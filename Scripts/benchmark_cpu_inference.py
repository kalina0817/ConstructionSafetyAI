import json
import random
import time
import platform
import statistics
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

ROOT = Path.cwd()
SAMPLE_SIZE = 30
WARMUP = 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

benchmarks = [
    {
        "dataset": "PPE",
        "images": ROOT / "data/standardized/construction-ppe/test/images",
        "models": {
            "YOLO11n": ROOT / "runs/detect/runs/detect/ppe_yolo11n_postcorrection_15ep/weights/best.pt",
            "RT-DETR-L": ROOT / "models/rtdetr_l_comparison/ConstructionSafetyAI_model_comparison/rtdetr_l_ppe_best.pt",
        },
    },
    {
        "dataset": "Toolset-35",
        "images": ROOT / "data/standardized/construction-toolset-35class/test/images",
        "models": {
            "YOLO11n": ROOT / "models/toolset35_final_model/toolset35_yolo11n_best.pt",
            "RT-DETR-L": ROOT / "models/rtdetr_l_comparison/ConstructionSafetyAI_model_comparison/rtdetr_l_toolset35_best.pt",
        },
    },
    {
        "dataset": "Vehicles",
        "images": ROOT / "data/standardized/construction-vehicle/test/images",
        "models": {
            "YOLO11n": ROOT / "runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt",
            "RT-DETR-L": ROOT / "models/rtdetr_l_comparison/ConstructionSafetyAI_model_comparison/rtdetr_l_vehicles_best.pt",
        },
    },
]

results = []

for dataset in benchmarks:
    image_dir = dataset["images"]
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    random.Random(42).shuffle(images)
    images = images[:SAMPLE_SIZE]

    print(f"\n{'=' * 65}")
    print(f"DATASET: {dataset['dataset']} | Images: {len(images)}")
    print(f"{'=' * 65}")

    for model_name, model_path in dataset["models"].items():
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"\nLoading {model_name}...")
        model = YOLO(str(model_path))

        print(f"Warming up with {WARMUP} images...")
        for image in images[:min(WARMUP, len(images))]:
            model.predict(
                source=str(image),
                imgsz=640,
                device="cpu",
                verbose=False,
            )

        timings = []
        print(f"Timing {len(images)} images...")

        for index, image in enumerate(images, start=1):
            start = time.perf_counter()
            model.predict(
                source=str(image),
                imgsz=640,
                device="cpu",
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)

            if index % 5 == 0 or index == len(images):
                print(f"  Completed {index}/{len(images)}")

        entry = {
            "dataset": dataset["dataset"],
            "model": model_name,
            "model_path": str(model_path),
            "images_tested": len(timings),
            "image_size": 640,
            "device": "cpu",
            "mean_ms_per_image": round(statistics.mean(timings), 2),
            "median_ms_per_image": round(statistics.median(timings), 2),
            "min_ms_per_image": round(min(timings), 2),
            "max_ms_per_image": round(max(timings), 2),
            "images_per_second": round(1000 / statistics.mean(timings), 3),
        }
        results.append(entry)

        print(
            f"  Mean: {entry['mean_ms_per_image']} ms/image | "
            f"Median: {entry['median_ms_per_image']} ms/image | "
            f"Speed: {entry['images_per_second']} images/sec"
        )

        del model

output_dir = ROOT / "reports/model_benchmark"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "cpu_results.json"

report = {
    "benchmark_date": datetime.now().isoformat(timespec="seconds"),
    "machine": platform.processor(),
    "platform": platform.platform(),
    "device": "CPU",
    "image_size": 640,
    "warmup_images": WARMUP,
    "results": results,
}

output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\n{'=' * 65}")
print(f"CPU BENCHMARK COMPLETE")
print(f"Results saved to: {output_file}")
print(f"{'=' * 65}")
