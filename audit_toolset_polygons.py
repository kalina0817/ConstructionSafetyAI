
from pathlib import Path
from collections import Counter

ROOT = Path(r"data\processed\construction-toolset-deduplicated")
config_path = ROOT / "data.yaml"

import yaml

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
names = config["names"]

polygon_counts = Counter()
box_counts = Counter()
empty_labels = 0
invalid_lines = []

for split in ["train", "valid", "test"]:
    label_dir = ROOT / split / "labels"

    for label in label_dir.glob("*.txt"):
        lines = label.read_text(encoding="utf-8").splitlines()

        if not any(line.strip() for line in lines):
            empty_labels += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            parts = line.split()

            try:
                class_id = int(parts[0])
                values = [float(v) for v in parts[1:]]
            except (ValueError, IndexError):
                invalid_lines.append(f"{label}:{line_number}")
                continue

            if not 0 <= class_id < len(names):
                invalid_lines.append(f"{label}:{line_number} invalid class")
                continue

            if len(values) == 4:
                box_counts[class_id] += 1
            elif len(values) >= 6 and len(values) % 2 == 0:
                polygon_counts[class_id] += 1
            else:
                invalid_lines.append(f"{label}:{line_number} unknown format")

print("\nANNOTATION FORMAT BY CLASS")
print("-" * 45)

for class_id, name in enumerate(names):
    boxes = box_counts[class_id]
    polygons = polygon_counts[class_id]
    print(
        f"{class_id:2}. {name}: "
        f"{boxes} boxes, {polygons} polygons"
    )

print("\nTotal bounding boxes:", sum(box_counts.values()))
print("Total polygons:", sum(polygon_counts.values()))
print("Empty label files:", empty_labels)
print("Invalid annotation lines:", len(invalid_lines))

if invalid_lines:
    print("\nFirst 20 issues:")
    for issue in invalid_lines[:20]:
        print(issue)

print("\nAudit only: no files were modified.")
