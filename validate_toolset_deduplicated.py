
from pathlib import Path
from collections import Counter
import hashlib
import yaml

ROOT = Path(r"data\processed\construction-toolset-deduplicated")
config_path = ROOT / "data.yaml"

if not config_path.exists():
    raise SystemExit(f"ERROR: Cannot find {config_path.resolve()}")

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
nc = config["nc"]

errors = []
image_hashes = {}
format_counts = Counter()
class_counts = Counter()
empty_labels = 0
images_checked = 0
label_lines = 0

for split in ["train", "valid", "test"]:
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        errors.append(f"Missing images or labels directory for {split}")
        continue

    for image in sorted(image_dir.iterdir()):
        if not image.is_file():
            continue

        images_checked += 1

        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in image_hashes:
            errors.append(
                f"Duplicate image: {image} and {image_hashes[digest]}"
            )
        else:
            image_hashes[digest] = image

        label = label_dir / f"{image.stem}.txt"

        if not label.is_file():
            errors.append(f"Missing label file: {label}")
            continue

        lines = label.read_text(encoding="utf-8").splitlines()
        if not any(line.strip() for line in lines):
            empty_labels += 1

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            label_lines += 1
            parts = line.split()

            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except (ValueError, IndexError):
                errors.append(f"Invalid annotation: {label}:{line_number}")
                continue

            if not 0 <= class_id < nc:
                errors.append(f"Invalid class ID: {label}:{line_number}")
                continue

            if not all(map(lambda v: abs(v) != float("inf") and v == v, values)):
                errors.append(f"Non-finite coordinate: {label}:{line_number}")
                continue

            class_counts[class_id] += 1

            if len(values) == 4:
                format_counts["bounding boxes"] += 1
                x, y, width, height = values

                if (
                    not all(0 <= v <= 1 for v in values)
                    or width <= 0
                    or height <= 0
                ):
                    errors.append(
                        f"Invalid box coordinates: {label}:{line_number}"
                    )

            elif len(values) >= 6 and len(values) % 2 == 0:
                format_counts["polygons"] += 1

                if not all(0 <= v <= 1 for v in values):
                    errors.append(
                        f"Invalid polygon coordinates: {label}:{line_number}"
                    )

            else:
                errors.append(
                    f"Unknown annotation format: {label}:{line_number}"
                )

print("\nTOOL SET DATASET VALIDATION")
print("-" * 35)
print("Images checked:", images_checked)
print("Annotation lines checked:", label_lines)
print("Annotation formats:", dict(format_counts))
print("Empty label files:", empty_labels)
print(f"Classes with annotations: {len(class_counts)} of {nc}")
print("Exact duplicate images remaining:", images_checked - len(image_hashes))
print("Validation errors:", len(errors))

if errors:
    print("\nFirst 30 errors:")
    for error in errors[:30]:
        print("ERROR:", error)
    print("\nRESULT: FAIL — review the errors.")
else:
    print("\nRESULT: PASS — structural checks passed.")

print("\nNo images or labels were modified by this script.")
print("No model training was started.")
