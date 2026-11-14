
from pathlib import Path
import hashlib

DATASET = Path(
    r"data\processed\construction-vehicle-detection-deduplicated"
)

CLASS_COUNT = 10
splits = ["train", "valid", "test"]
seen_hashes = {}
duplicates = []
errors = []
image_total = 0
annotation_total = 0

for split in splits:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    for image in image_dir.iterdir():
        if not image.is_file():
            continue

        image_total += 1
        digest = hashlib.sha256(image.read_bytes()).hexdigest()

        if digest in seen_hashes:
            duplicates.append(
                (str(image), seen_hashes[digest])
            )
        else:
            seen_hashes[digest] = str(image)

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            errors.append(f"Missing label: {label}")
            continue

        for line_number, line in enumerate(
            label.read_text().splitlines(), start=1
        ):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label}:{line_number}: expected 5 values"
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, width, height = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{label}:{line_number}: invalid number"
                )
                continue

            if not 0 <= class_id < CLASS_COUNT:
                errors.append(
                    f"{label}:{line_number}: invalid class ID"
                )

            if not all(
                0 <= value <= 1
                for value in (x, y, width, height)
            ) or width <= 0 or height <= 0:
                errors.append(
                    f"{label}:{line_number}: invalid bounding box"
                )

            annotation_total += 1

print("\n--- FINAL DATASET VALIDATION ---")
print("Images checked:", image_total)
print("Exact duplicates remaining:", len(duplicates))
print("Annotation rows checked:", annotation_total)
print("Validation errors:", len(errors))

for error in errors[:20]:
    print("ERROR:", error)

for duplicate in duplicates[:10]:
    print("DUPLICATE:", duplicate)

passed = (
    image_total == 17794
    and not duplicates
    and not errors
)

print("RESULT:", "PASS" if passed else "REVIEW REQUIRED")
