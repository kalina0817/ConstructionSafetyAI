import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path.cwd()

DATASETS = {
    "ppe": {
        "folder": "data/processed/construction-ppe",
        "yolo": "runs/detect/runs/detect/ppe_yolo11n_postcorrection_15ep/weights/best.pt",
        "rtdetr": "models/rtdetr_l_comparison/ConstructionSafetyAI_model_comparison/rtdetr_l_ppe_best.pt",
    },
    "toolset": {
        "folder": "data/processed/construction-toolset",
        "yolo": "models/toolset35_final_model/toolset35_yolo11n_best.pt",
        "rtdetr": "models/rtdetr_l_comparison/ConstructionSafetyAI_model_comparison/rtdetr_l_toolset35_best.pt",
    },
    "vehicles": {
        "folder": "data/processed/construction-vehicle",
        "yolo": "runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt",
        "rtdetr": "models/rtdetr_l_comparison/ConstructionSafetyAI_model_comparison/rtdetr_l_vehicles_best.pt",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=DATASETS, required=True)
args = parser.parse_args()

config = DATASETS[args.dataset]
dataset_root = ROOT / config["folder"]

output_dir = ROOT / "reports" / "model_benchmarks" / datetime.now().strftime("%Y%m%d_%H%M%S") / args.dataset
output_dir.mkdir(parents=True, exist_ok=True)

results = []

for model_type in ("yolo", "rtdetr"):
    checkpoint = ROOT / config[model_type]

    print(f"\nLoading {model_type.upper()}: {checkpoint}", flush=True)
    model = YOLO(str(checkpoint))

    names = model.names
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    names = {int(i): str(name) for i, name in names.items()}

    data_config = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": [names[i] for i in range(len(names))],
    }

    yaml_path = output_dir / f"{model_type}_data.yaml"
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data_config, file, sort_keys=False)

    start = time.perf_counter()
    metrics = model.val(
        data=str(yaml_path),
        split="val",
        imgsz=640,
        batch=1,
        device="cpu",
        workers=0,
        plots=False,
        verbose=False,
        project=str(output_dir / "runs"),
        name=model_type,
        exist_ok=True,
    )
    elapsed = time.perf_counter() - start

    row = {
        "dataset": args.dataset,
        "model": model_type.upper(),
        "checkpoint": str(checkpoint),
        "classes": len(names),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "total_evaluation_seconds": round(elapsed, 2),
    }
    results.append(row)

    print(json.dumps(row, indent=2), flush=True)

with (output_dir / "benchmark_results.json").open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print(f"\nSaved benchmark results to: {output_dir}", flush=True)
