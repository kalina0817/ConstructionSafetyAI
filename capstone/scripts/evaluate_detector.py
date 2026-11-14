
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from construction_safety.config import (
    EVALUATION_ROOT,
    MODEL_SPECS,
    PROJECT_ROOT,
    WORKSPACE_ROOT,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _number(value):
    if hasattr(value, "item"):
        value = value.item()
    return float(value) if isinstance(value, (int, float)) else str(value)


def run_evaluation(
    model_name,
    split="val",
    allow_locked_test=False,
    image_size=640,
    batch_size=4,
    device="cpu",
):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown detector: {model_name}")

    if split not in {"val", "test"}:
        raise ValueError("Split must be val or test.")

    if split == "test" and not allow_locked_test:
        raise ValueError(
            "The detector test split is locked. Explicit confirmation is "
            "required for the preregistered final evaluation."
        )

    if image_size <= 0 or batch_size <= 0:
        raise ValueError("Image size and batch size must be positive.")

    spec = MODEL_SPECS[model_name]

    checkpoint = WORKSPACE_ROOT / spec["checkpoint"]
    dataset_path = WORKSPACE_ROOT / spec["dataset"]

    # Existing detectors use training_args; Toolset-35 uses
    # training_metadata. Do not fabricate an args.yaml file.
    if spec.get("training_args"):
        provenance_key = "training_args"
    elif spec.get("training_metadata"):
        provenance_key = "training_metadata"
    else:
        raise ValueError(
            f"No training provenance configured for detector: {model_name}"
        )

    training_provenance = WORKSPACE_ROOT / spec[provenance_key]

    for path in (checkpoint, dataset_path, training_provenance):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    with dataset_path.open("r", encoding="utf-8-sig") as file:
        dataset = yaml.safe_load(file)

    if not isinstance(dataset, dict) or not dataset.get("test"):
        raise ValueError(f"Dataset YAML has no test split: {dataset_path}")

    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = EVALUATION_ROOT / f"{timestamp}_{model_name}_{split}"
    output_dir.mkdir(parents=False, exist_ok=False)

    model = YOLO(str(checkpoint))

    result = model.val(
        data=str(dataset_path),
        split=split,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=0,
        plots=False,
        save_json=False,
        project=str(output_dir),
        name="ultralytics",
        exist_ok=False,
    )

    class_names = model.names
    per_class = {}

    for class_id, value in enumerate(result.box.maps):
        name = (
            class_names.get(class_id, str(class_id))
            if isinstance(class_names, dict)
            else class_names[class_id]
        )
        per_class[str(name)] = float(value)

    provenance_path = str(training_provenance)
    provenance_sha256 = sha256_file(training_provenance)

    artifacts = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_yaml": str(dataset_path),
        "dataset_yaml_sha256": sha256_file(dataset_path),
        "training_provenance": provenance_path,
        "training_provenance_type": provenance_key,
        "training_provenance_sha256": provenance_sha256,
    }

    # Preserve the original artifact fields for existing detectors.
    if provenance_key == "training_args":
        artifacts["training_args"] = provenance_path
        artifacts["training_args_sha256"] = provenance_sha256
    else:
        artifacts["training_metadata"] = provenance_path
        artifacts["training_metadata_sha256"] = provenance_sha256

    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector": model_name,
        "evaluation_split": split,
        "test_split_was_locked": split == "test",
        "model_selection_or_threshold_tuning_on_split": False,
        "configuration": {
            "image_size": image_size,
            "batch_size": batch_size,
            "device": device,
            "workers": 0,
        },
        "artifacts": artifacts,
        "dataset_classes": dataset.get("names"),
        "metrics": {
            key: _number(value)
            for key, value in result.results_dict.items()
        },
        "per_class_map50_95": per_class,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "ultralytics": _package_version("ultralytics"),
            "numpy": _package_version("numpy"),
        },
        "limitations": [
            "Detector metrics do not evaluate contextual safety findings.",
            "The test split must not be used for model selection or tuning.",
            "Results depend on the exact dataset, checkpoint, and evaluator.",
        ],
    }

    report_path = output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report_path, report


def main():
    parser = argparse.ArgumentParser(
        description="Record reproducible validation metrics for a detector."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
    )
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    try:
        path, report = run_evaluation(
            args.model,
            args.split,
            args.allow_locked_test,
            args.image_size,
            args.batch_size,
            args.device,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Evaluation written to: {path}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
