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


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "reports" / "detector_evaluations"

BASELINE_SPECS = {
    "ppe": {
        "checkpoint": Path(
            "runs/detect/runs/detect/"
            "ppe_yolo11n_postcorrection_15ep/weights/best.pt"
        ),
        "dataset": Path("data/processed/construction-ppe-corrected/data.yaml"),
        "training_args": Path(
            "runs/detect/runs/detect/"
            "ppe_yolo11n_postcorrection_15ep/args.yaml"
        ),
    },
    "toolset": {
        "checkpoint": Path(
            "runs/detect/runs/detect/"
            "toolset_yolo11n_continuation_final5-2/weights/best.pt"
        ),
        "dataset": Path(
            "data/processed/construction-toolset-safety-clean/data.yaml"
        ),
        "training_args": Path(
            "runs/detect/runs/detect/"
            "toolset_yolo11n_continuation_final5-2/args.yaml"
        ),
    },
    "vehicles": {
        "checkpoint": Path(
            "runs/detect/runs/detect/"
            "vehicle_yolo11n_continuation_5ep/weights/best.pt"
        ),
        "dataset": Path(
            "data/processed/"
            "construction-vehicle-detection-deduplicated/data.yaml"
        ),
        "training_args": Path(
            "runs/detect/runs/detect/"
            "vehicle_yolo11n_continuation_5ep/args.yaml"
        ),
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def load_dataset_config(path):
    with Path(path).open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML is invalid: {path}")
    if not config.get("test"):
        raise ValueError(f"Dataset YAML has no test split: {path}")
    return config


def run_evaluation(
    model_name,
    split="val",
    allow_locked_test=False,
    imgsz=640,
    batch=4,
    device="cpu",
):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be 'val' or 'test'.")
    if split == "test" and not allow_locked_test:
        raise ValueError(
            "The detector test split is locked. Use --allow-locked-test only "
            "for the preregistered final evaluation after model selection "
            "and threshold tuning are complete."
        )
    if model_name not in BASELINE_SPECS:
        raise ValueError(f"Unknown detector baseline: {model_name}")
    if imgsz <= 0 or batch <= 0:
        raise ValueError("Image size and batch size must be positive.")

    spec = BASELINE_SPECS[model_name]
    checkpoint = ROOT / spec["checkpoint"]
    dataset_path = ROOT / spec["dataset"]
    training_args = ROOT / spec["training_args"]
    for path in (checkpoint, dataset_path, training_args):
        if not path.is_file():
            raise FileNotFoundError(f"Required baseline file not found: {path}")
    dataset_config = load_dataset_config(dataset_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{timestamp}_{model_name}_{split}"
    evaluation_root = OUTPUT_ROOT / run_name
    evaluation_root.mkdir(parents=True, exist_ok=False)

    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(dataset_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        plots=False,
        save_json=False,
        project=str(evaluation_root),
        name="ultralytics_validation",
        exist_ok=False,
    )

    metric_values = {
        key: _as_builtin_number(value)
        for key, value in metrics.results_dict.items()
    }
    per_class_ap = {}
    class_names = model.names
    for class_id, average_precision in enumerate(metrics.box.maps):
        class_name = (
            class_names.get(class_id, str(class_id))
            if isinstance(class_names, dict)
            else class_names[class_id]
        )
        per_class_ap[str(class_name)] = float(average_precision)

    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector": model_name,
        "evaluation_split": split,
        "test_split_was_locked": split == "test",
        "model_selection_or_threshold_tuning_on_this_split": False,
        "configuration": {
            "image_size": imgsz,
            "batch_size": batch,
            "device": device,
            "workers": 0,
        },
        "artifacts": {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dataset_yaml": str(dataset_path.relative_to(ROOT)),
            "dataset_yaml_sha256": sha256_file(dataset_path),
            "training_args": str(training_args.relative_to(ROOT)),
            "training_args_sha256": sha256_file(training_args),
        },
        "dataset_classes": dataset_config.get("names"),
        "metrics": metric_values,
        "per_class_map50_95": per_class_ap,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "ultralytics": installed_version("ultralytics"),
            "numpy": installed_version("numpy"),
        },
        "limitations": [
            "Detector metrics do not evaluate contextual safety findings.",
            "Do not use the test split for model or threshold selection.",
            "A reproducible run may differ from previously reported values "
            "if source data, checkpoints, package versions, or evaluator "
            "settings differ.",
        ],
    }
    report_path = evaluation_root / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report_path, report


def _as_builtin_number(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce a selected detector baseline evaluation and record "
            "checkpoint and environment provenance."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(BASELINE_SPECS),
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help=(
            "Defaults to validation data. The test split is locked and "
            "requires an explicit --allow-locked-test flag."
        ),
    )
    parser.add_argument(
        "--allow-locked-test",
        action="store_true",
        help=(
            "Permit one final test evaluation after model selection and "
            "thresholds are frozen."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        report_path, report = run_evaluation(
            model_name=args.model,
            split=args.split,
            allow_locked_test=args.allow_locked_test,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Evaluation report written to: {report_path}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
