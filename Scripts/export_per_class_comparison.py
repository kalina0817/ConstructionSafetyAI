import json
import csv
from pathlib import Path

runs = {
    "ppe": ("20261122_002751", 11),
    "toolset": ("20261122_003030", 35),
    "vehicles": ("20261122_004042", 10),
}

rows = []

for dataset, (timestamp, expected) in runs.items():
    folder = Path("reports/model_benchmarks") / timestamp / "test" / dataset

    for model, filename in [
        ("YOLO11n", "yolo_per_class_metrics.json"),
        ("RT-DETR-L", "rtdetr_per_class_metrics.json"),
    ]:
        path = folder / filename

        with path.open(encoding="utf-8-sig") as f:
            payload = json.load(f)

        classes = payload if isinstance(payload, list) else payload["value"]

        if len(classes) != expected:
            raise ValueError(
                f"{path}: expected {expected} classes, found {len(classes)}"
            )

        for item in classes:
            rows.append({
                "dataset": dataset,
                "model": model,
                **item,
            })

output = Path("reports/model_benchmarks/per_class_test_comparison.csv")
output.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "dataset", "model", "class_id", "class_name",
    "ground_truth_present", "precision", "recall", "map50", "map50_95"
]

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {output}")
print(f"Rows: {len(rows)}")
print("Expected: 112 (56 classes x 2 models)")
