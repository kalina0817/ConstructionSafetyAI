
from pathlib import Path
from collections import Counter
import hashlib
import math
import yaml

ROOT = Path(
    r"data\processed\construction-toolset-deduplicated-detection"
)

config_path = ROOT / "data.yaml"
if not config_path.is_file():
    raise SystemExit(f"ERROR: Missing {config_path}")

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
names = config.get("names", [])
nc = config.get("nc")

errors = []
hashes = {}
formats = Counter()
class_counts = Counter()
images_checked = 0
labels_checked = 0
empty_labels = 0
annotation_lines = 0

if not isinstance(nc, int) or len(names) != nc:
    errors.append("Class count does not match class names")

for split in ("train", "valid", "test"):
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        errors.append(f"Missing image or label folder for {split}")
        continue

    split_images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    for image in split_images:
        images_checked += 1
        digest = hashlib.sha256(image.read_bytes()).hexdigest()

        if digest in hashes:
            errors.append(
                f"Duplicate image: {image} and {hashes[digest]}"
            )
        else:
            hashes[digest] = image

        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            errors.append(f"Missing label: {label}")
            continue

        labels_checked += 1
        lines = label.read_text(encoding="utf-8").splitlines()

        if not any(line.strip() for line in lines):
            empty_labels += 1

        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            annotation_lines += 1
            parts = line.split()

            try:
                class_value = float(parts[0])
                values = [float(v) for v in parts[1:]]
            except (ValueError, IndexError):
                errors.append(f"Invalid values: {label}:{line_no}")
                continue

            if not math.isfinite(class_value) or not class_value.is_integer():
                errors.append(f"Invalid class ID: {label}:{line_no}")
                continue

            class_id = int(class_value)
            if not 0 <= class_id < nc:
                errors.append(f"Class ID out of range: {label}:{line_no}")
                continue

            if len(values) != 4:
                errors.append(
                    f"Not a YOLO bounding box (expected 4 coordinates): "
                    f"{label}:{line_no}"
                )
                continue

            if not all(math.isfinite(v) for v in values):
                errors.append(f"Non-finite coordinate: {label}:{line_no}")
                continue

            x, y, width, height = values

            if not all(0 <= v <= 1 for v in values):
                errors.append(f"Coordinate outside 0-1: {label}:{line_no}")

            if width <= 0 or height <= 0:
                errors.append(f"Non-positive box size: {label}:{line_no}")

            if (
                x - width / 2 < -1e-6
                or x + width / 2 > 1 + 1e-6
                or y - height / 2 < -1e-6
                or y + height / 2 > 1 + 1e-6
            ):
                errors.append(f"Bounding box exceeds image: {label}:{line_no}")

            formats["bounding boxes"] += 1
            class_counts[class_id] += 1

print("\nCONVERTED TOOLSET DATASET VALIDATION")
print("-" * 45)
print("Images checked:", images_checked)
print("Label files checked:", labels_checked)
print("Annotation lines checked:", annotation_lines)
print("Annotation formats:", dict(formats))
print("Empty label files:", empty_labels)
print(f"Classes with annotations: {len(class_counts)} of {nc}")
print("Exact duplicate images remaining:", images_checked - len(hashes))
print("Validation errors:", len(errors))

if errors:
    print("\nFirst 30 errors:")
    for error in errors[:30]:
        print("ERROR:", error)
    print("\nRESULT: FAIL — do not train yet.")
else:
    print("\nRESULT: PASS — structural checks passed.")

print("No files were modified. No training was started.")
