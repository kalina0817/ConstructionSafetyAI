import json
from pathlib import Path

root = Path.cwd()
folder = root / "reports" / "model_benchmark"

gpu = json.loads((folder / "colab_results.json").read_text(encoding="utf-8"))
cpu = json.loads((folder / "cpu_results.json").read_text(encoding="utf-8"))

cpu_lookup = {
    (item["dataset"], item["model"]): item
    for item in cpu["results"]
}

recommendations = {
    "PPE": (
        "YOLO11n for default CPU screening. RT-DETR-L is an accuracy-oriented "
        "alternative when its additional detection accuracy justifies the latency."
    ),
    "Toolset-35": (
        "YOLO11n. It has higher recall and mAP, as well as substantially "
        "faster CPU inference. RT-DETR-L has higher precision but lower recall."
    ),
    "Vehicles": (
        "YOLO11n for default CPU screening. RT-DETR-L is an accuracy-oriented "
        "alternative when its higher recall and mAP justify the latency."
    ),
}

combined = []

for item in gpu:
    key = (item["dataset"], item["model"])
    cpu_item = cpu_lookup[key]

    combined.append({
        "dataset": item["dataset"],
        "model": item["model"],
        "gpu_validation": {
            "precision": item["precision"],
            "recall": item["recall"],
            "mAP50": item["mAP50"],
            "mAP50_95": item["mAP50_95"],
            "inference_ms": item["inference_ms"],
            "elapsed_seconds": item["elapsed_seconds"],
        },
        "local_cpu": {
            "images_tested": cpu_item["images_tested"],
            "image_size": cpu_item["image_size"],
            "mean_ms_per_image": cpu_item["mean_ms_per_image"],
            "median_ms_per_image": cpu_item["median_ms_per_image"],
            "min_ms_per_image": cpu_item["min_ms_per_image"],
            "max_ms_per_image": cpu_item["max_ms_per_image"],
            "images_per_second": cpu_item["images_per_second"],
        },
    })

for dataset in ["PPE", "Toolset-35", "Vehicles"]:
    yolo = next(x for x in combined if x["dataset"] == dataset and x["model"] == "YOLO11n")
    rtdetr = next(x for x in combined if x["dataset"] == dataset and x["model"] == "RT-DETR-L")
    yolo["cpu_speedup_over_rtdetr"] = round(
        rtdetr["local_cpu"]["mean_ms_per_image"] /
        yolo["local_cpu"]["mean_ms_per_image"], 2
    )
    rtdetr["cpu_speedup_over_rtdetr"] = 1.0

report = {
    "title": "ConstructionSafetyAI Model Comparison",
    "gpu_benchmark": {
        "hardware": "Google Colab Tesla T4",
        "metrics": "Validation precision, recall, mAP50, mAP50-95 and inference time",
    },
    "cpu_benchmark": {
        "hardware": cpu.get("machine", "Not recorded"),
        "platform": cpu.get("platform", "Not recorded"),
        "device": "CPU",
        "image_size": cpu.get("image_size", 640),
        "images_per_model_per_dataset": 30,
        "warmup_images": cpu.get("warmup_images", 2),
        "timing_method": "Per-image Ultralytics predict() wall-clock time, including preprocessing and postprocessing",
    },
    "results": combined,
    "recommendations": recommendations,
    "limitations": [
        "GPU validation inference timings and local CPU predict() wall-clock timings use different measurement procedures and should not be directly compared.",
        "CPU timings are based on 30 sampled test images per model and dataset.",
        "Detection metrics describe performance on the evaluated datasets; they do not establish real-world hazard confirmation or safety certification.",
    ],
}

json_path = folder / "combined_model_comparison.json"
json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

lines = [
    "# ConstructionSafetyAI Model Comparison",
    "",
    "## Benchmark setup",
    "",
    "- GPU evaluation: Google Colab Tesla T4.",
    "- CPU evaluation: local Windows machine, CPU-only PyTorch.",
    "- Input size: 640 × 640.",
    "- CPU sample: 30 test images per model and dataset, with 2 warm-up images.",
    "- CPU timing measures the full per-image `predict()` call.",
    "",
    "GPU and CPU timings are reported separately because their measurement procedures differ.",
    "",
    "## Results",
    "",
    "| Dataset | Model | Precision | Recall | mAP50 | mAP50–95 | GPU inference (ms) | CPU mean (ms/image) | CPU median (ms/image) | CPU images/sec |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for item in combined:
    g = item["gpu_validation"]
    c = item["local_cpu"]
    lines.append(
        f"| {item['dataset']} | {item['model']} "
        f"| {g['precision']:.4f} | {g['recall']:.4f} "
        f"| {g['mAP50']:.4f} | {g['mAP50_95']:.4f} "
        f"| {g['inference_ms']:.2f} | {c['mean_ms_per_image']:.2f} "
        f"| {c['median_ms_per_image']:.2f} | {c['images_per_second']:.3f} |"
    )

lines += [
    "",
    "## CPU speed comparison",
    "",
    "| Dataset | YOLO11n mean (ms) | RT-DETR-L mean (ms) | YOLO11n speed advantage |",
    "|---|---:|---:|---:|",
]

for dataset in ["PPE", "Toolset-35", "Vehicles"]:
    y = next(x for x in combined if x["dataset"] == dataset and x["model"] == "YOLO11n")
    r = next(x for x in combined if x["dataset"] == dataset and x["model"] == "RT-DETR-L")
    lines.append(
        f"| {dataset} | {y['local_cpu']['mean_ms_per_image']:.2f} "
        f"| {r['local_cpu']['mean_ms_per_image']:.2f} "
        f"| {y['cpu_speedup_over_rtdetr']:.2f}× |"
    )

lines += ["", "## Model recommendations", ""]

for dataset, recommendation in recommendations.items():
    lines.append(f"### {dataset}")
    lines.append("")
    lines.append(recommendation)
    lines.append("")

lines += [
    "## Interpretation and limitations",
    "",
    "- PPE: RT-DETR-L has higher precision, recall, mAP50, and mAP50–95. YOLO11n is substantially faster on the local CPU.",
    "- Vehicles: RT-DETR-L has higher precision, recall, mAP50, and mAP50–95. YOLO11n is substantially faster on the local CPU.",
    "- Toolset-35: YOLO11n has higher recall, mAP50, and mAP50–95. RT-DETR-L has higher precision.",
    "- CPU timing includes preprocessing and postprocessing, not only neural-network execution.",
    "- CPU timing uses a 30-image sample and should be treated as a practical local benchmark, not a complete hardware-independent performance claim.",
    "- Detection results are screening evidence, not proof that a hazard is present or absent. The system is not a safety certification tool.",
    "",
    "## Source files",
    "",
    "- `colab_results.json`",
    "- `cpu_results.json`",
    "",
]

md_path = folder / "combined_model_comparison.md"
md_path.write_text("\n".join(lines), encoding="utf-8")

print("Combined report created:")
print(json_path)
print(md_path)
print("\nCPU speed advantage of YOLO11n:")
for dataset in ["PPE", "Toolset-35", "Vehicles"]:
    y = next(x for x in combined if x["dataset"] == dataset and x["model"] == "YOLO11n")
    print(f"{dataset}: {y['cpu_speedup_over_rtdetr']:.2f}x")
