import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from construction_safety.config import (
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


def train_detector(
    model_name,
    initial_weights,
    epochs,
    image_size=640,
    batch_size=4,
    seed=42,
    device="cpu",
):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown detector: {model_name}")
    if epochs <= 0 or image_size <= 0 or batch_size <= 0:
        raise ValueError("Epochs, image size, and batch size must be positive.")
    initial_weights = Path(initial_weights)
    if not initial_weights.is_absolute():
        initial_weights = (WORKSPACE_ROOT / initial_weights).resolve()
    if not initial_weights.is_file():
        raise FileNotFoundError(
            f"Initial model weights not found: {initial_weights}"
        )

    dataset_path = WORKSPACE_ROOT / MODEL_SPECS[model_name]["dataset"]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8-sig") as file:
        dataset = yaml.safe_load(file)
    if not isinstance(dataset, dict) or not dataset.get("train"):
        raise ValueError(f"Dataset YAML has no train split: {dataset_path}")

    train_root = PROJECT_ROOT / "reports" / "training_runs"
    train_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{model_name}_seed{seed}_{epochs}ep"
    model = YOLO(str(initial_weights))
    model.train(
        data=str(dataset_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=0,
        seed=seed,
        deterministic=True,
        val=True,
        split="val",
        project=str(train_root),
        name=run_name,
        exist_ok=False,
    )
    run_dir = Path(model.trainer.save_dir)
    provenance = {
        "detector": model_name,
        "initial_weights": str(initial_weights),
        "initial_weights_sha256": sha256_file(initial_weights),
        "dataset_yaml": str(dataset_path),
        "dataset_yaml_sha256": sha256_file(dataset_path),
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "deterministic": True,
        "training_split": "train",
        "validation_split": "val",
        "test_split_used": False,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    (run_dir / "capstone_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a detector on its configured training split. "
            "This command does not evaluate the locked test split."
        )
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        result = train_detector(
            args.model,
            args.initial_weights,
            args.epochs,
            args.image_size,
            args.batch_size,
            args.seed,
            args.device,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Training completed: {result}")


if __name__ == "__main__":
    main()
