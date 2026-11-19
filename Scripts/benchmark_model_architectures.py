import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path.cwd()
OUTPUT = ROOT / "reports" / "model_benchmarks"
CONFIGS = OUTPUT / "dataset_configs"
OUTPUT.mkdir(parents=True, exist_ok=True)
CONFIGS.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PPE": ROOT / "data" / "processed" / "construction-ppe",
    "Toolset35": ROOT / "data" / "processed" / "construction-toolset",
    "Vehicles": ROOT / "data" / "processed" / "construction-vehicle",
}

MODELS = {
    "PPE": {
        "YOLO11n": ROOT / "runs" / "detect" / "runs" / "detect" / "ppe_yolo11n_postcorrection_15ep" / "weights" / "best.pt",
        "RT-DETR-L": ROOT / "models" / "rtdetr_l_comparison" / "ConstructionSafetyAI_model_comparison" / "rtdetr_l_ppe_best.pt",
    },
    "Toolset35": {
        "YOLO11n": ROOT / "models" / "toolset35_final_model" / "toolset35_yolo11n_best.pt",
        "RT-DETR-L": ROOT / "models" / "rtdetr_l_comparison" / "ConstructionSafetyAI_model_comparison" / "rtdetr_l_toolset35_best.pt",
    },
    "Vehicles": {
        "YOLO11n": ROOT / "runs" / "detect" / "runs" / "detect" / "vehicle_yolo11n_corrected_dataset_5ep" / "weights" / "best.pt",
        "RT-DETR-L": ROOT / "models" / "rtdetr_l_comparison" / "ConstructionSafetyAI_model_comparison" / "rtdetr_l_vehicles_best.pt",
    },
}

records = []

for dataset_name, dataset_root in DATASETS.items():
    print(f"\n{'=' * 65}")
    print(f"DATASET: {dataset_name}")
    print(f"Validation images: {len(list((dataset_root / 'valid' / 'images').glob('*')))}")

    for model_name, checkpoint in MODELS[dataset_name].items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        print(f"\nEvaluating {model_name}: {checkpoint.name}", flush=True)
        model = YOLO(str(checkpoint))

        names = model.names
        names = [names[i] for i in range(len(names))]

        data_config = {
            "path": str(dataset_root.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(names),
            "names": names,
        }

        config_path = CONFIGS / f"{dataset_name.lower()}_35_or_dataset.yaml"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data_config, file, sort_keys=False, allow_unicode=True)

        start = time.perf_counter()
        metrics = model.val(
            data=str(config_path),
            split="val",
            imgsz=640,
            batch=1,
            device="cpu",
            workers=0,
            conf=0.001,
            iou=0.7,
            half=False,
            plots=False,
            verbose=False,
            project=str(OUTPUT / "runs"),
            name=f"{dataset_name}_{model_name.replace('-', '').lower()}",
            exist_ok=True,
        )
        elapsed = time.perf_counter() - start

        box = metrics.box
        speed = metrics.speed

        record = {
            "dataset": dataset_name,
            "model": model_name,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "validation_images": len(list((dataset_root / "valid" / "images").glob("*"))),
            "classes": len(names),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "preprocess_ms_per_image": float(speed["preprocess"]),
            "inference_ms_per_image": float(speed["inference"]),
            "postprocess_ms_per_image": float(speed["postprocess"]),
            "evaluation_elapsed_seconds": round(elapsed, 2),
        }
        records.append(record)

        print(
            f"Precision: {record['precision']:.4f} | "
            f"Recall: {record['recall']:.4f} | "
            f"mAP50: {record['mAP50']:.4f} | "
            f"mAP50-95: {record['mAP50_95']:.4f}"
        )
        print(
            f"Inference: {record['inference_ms_per_image']:.2f} ms/image",
            flush=True,
        )

        del model

json_path = OUTPUT / "benchmark_results.json"
with json_path.open("w", encoding="utf-8") as file:
    json.dump(records, file, indent=2)

csv_path = OUTPUT / "benchmark_results.csv"
import csv

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"\nBenchmark complete.")
print(f"JSON: {json_path}")
print(f"CSV:  {csv_path}")
