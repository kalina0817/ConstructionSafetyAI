
from pathlib import Path
from collections import Counter
import hashlib
import math
import yaml

ROOT = Path(r"data\raw\construction-ppe")
config = yaml.safe_load(
    (ROOT / "data.yaml").read_text(encoding="utf-8")
)

nc = config["nc"]
names = config["names"]
errors = []
image_hashes = {}
class_counts = Counter()
format_counts = Counter()
images_checked = 0
label_lines = 0
empty_labels = 0
missing_labels = 0
label_files_seen = set()

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ("train", "valid", "test"):
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        errors.append(f"Missing images or labels folder: {split}")
        continue

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print(f"\n{split.upper()}: {len(images)} images")

    for image in images:
        images_checked += 1
        digest = hashlib.sha256(image.read_bytes()).hexdigest()

        if digest in image_hashes:
            errors.append(
                f"Duplicate image across dataset: {image} "
                f"matches {image_hashes[digest]}"
            )
        else:
            image_hashes[digest] = image

        label = label_dir / f"{image.stem}.txt"

        if not label.is_file():
            missing_labels += 1
            errors.append(f"Missing label: {label}")
            continue

        label_files_seen.add(label.resolve())
        lines = label.read_text(encoding="utf-8").splitlines()
        nonempty = [line for line in lines if line.strip()]

        if not nonempty:
            empty_labels += 1

        for line_no, line in enumerate(nonempty, start=1):
            label_lines += 1
            parts = line.split()

            try:
                class_value = float(parts[0])
                values = [float(v) for v in parts[1:]]
            except (ValueError, IndexError):
                errors.append(f"Invalid numeric values: {label}:{line_no}")
                continue

            if not math.isfinite(class_value) or not class_value.is_integer():
                errors.append(f"Invalid class ID: {label}:{line_no}")
                continue

            class_id = int(class_value)
            if not 0 <= class_id < nc:
                errors.append(f"Class ID out of range: {label}:{line_no}")
                continue

            if not all(math.isfinite(v) for v in values):
                errors.append(f"Non-finite coordinate: {label}:{line_no}")
                continue

            if len(values) == 4:
                format_counts["boxes"] += 1
                x, y, width, height = values

                if not all(0 <= v <= 1 for v in values):
                    errors.append(f"Box coordinate outside 0–1: {label}:{line_no}")

                if width <= 0 or height <= 0:
                    errors.append(f"Invalid box size: {label}:{line_no}")

                if (
                    x - width / 2 < -1e-6
                    or x + width / 2 > 1 + 1e-6
                    or y - height / 2 < -1e-6
                    or y + height / 2 > 1 + 1e-6
                ):
                    errors.append(f"Box extends beyond image: {label}:{line_no}")

            elif len(values) >= 6 and len(values) % 2 == 0:
                format_counts["polygons"] += 1

                if not all(0 <= v <= 1 for v in values):
                    errors.append(
                        f"Polygon coordinate outside 0–1: {label}:{line_no}"
                    )
            else:
                errors.append(f"Unknown annotation format: {label}:{line_no}")

            class_counts[class_id] += 1

    actual_labels = {
        p.resolve() for p in label_dir.glob("*.txt") if p.is_file()
    }
    extra_labels = actual_labels - label_files_seen
    if extra_labels:
        print(f"Label files not matched to images in {split}: {len(extra_labels)}")

print("\nCONSTRUCTION-PPE AUDIT")
print("-" * 40)
print("Images checked:", images_checked)
print("Annotation lines checked:", label_lines)
print("Annotation formats:", dict(format_counts))
print("Empty label files:", empty_labels)
print("Missing label files:", missing_labels)
print("Classes with annotations:", len(class_counts), "of", nc)
print("Exact duplicate images detected:", images_checked - len(image_hashes))
print("Validation errors:", len(errors))

print("\nCLASS COUNTS")
for class_id, name in enumerate(names):
    print(f"{class_id}: {name}: {class_counts[class_id]} annotations")

if errors:
    print("\nFirst 30 issues:")
    for error in errors[:30]:
        print("ERROR:", error)

print("\nAudit only: no files were modified.")
