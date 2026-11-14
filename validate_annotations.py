from pathlib import Path
import math
import sys

import yaml


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "processed" / "construction-ppe-corrected"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def load_class_count(config_path):
    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    class_count = config.get("nc")
    class_names = config.get("names")
    if not isinstance(class_count, int) or not isinstance(
        class_names, (list, dict)
    ):
        raise ValueError(f"Invalid class configuration: {config_path}")
    if len(class_names) != class_count:
        raise ValueError(
            f"Class count mismatch in {config_path}: "
            f"nc={class_count}, names={len(class_names)}"
        )
    return class_count


def validate_annotation(label_file, class_count):
    errors = []
    annotation_count = 0

    for line_number, line in enumerate(
        label_file.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        annotation_count += 1
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("Expected 5 values")

            class_value = float(parts[0])
            x, y, width, height = map(float, parts[1:])
            if not math.isfinite(class_value) or not class_value.is_integer():
                raise ValueError("Class ID must be a finite integer")
            if not all(math.isfinite(value) for value in (x, y, width, height)):
                raise ValueError("Box coordinates must be finite")

            class_id = int(class_value)
            if not 0 <= class_id < class_count:
                raise ValueError(f"Invalid class ID: {class_id}")
            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                raise ValueError("Invalid normalized box coordinates")
            if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
                raise ValueError("Box extends beyond image horizontally")
            if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
                raise ValueError("Box extends beyond image vertically")
        except (ValueError, OverflowError) as error:
            errors.append(
                f"{label_file} line {line_number}: {error} | {line}"
            )

    return annotation_count, errors


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    class_count = load_class_count(DATASET / "data.yaml")
    errors = []
    files_checked = 0
    annotations_checked = 0

    for split in SPLITS:
        image_dir = DATASET / split / "images"
        label_dir = DATASET / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing images or labels directory for split: {split}"
            )

        image_stems = {
            path.stem
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        label_files = {
            path.stem: path
            for path in label_dir.glob("*.txt")
            if path.is_file()
        }

        errors.extend(
            f"{split}: image has no label: {stem}"
            for stem in sorted(image_stems - label_files.keys())
        )
        errors.extend(
            f"{split}: label has no image: {path}"
            for stem, path in sorted(label_files.items())
            if stem not in image_stems
        )

        for label_file in label_files.values():
            files_checked += 1
            count, annotation_errors = validate_annotation(
                label_file, class_count
            )
            annotations_checked += count
            errors.extend(annotation_errors)

    print("Dataset:", DATASET)
    print("Classes:", class_count)
    print("Label files checked:", files_checked)
    print("Annotations checked:", annotations_checked)
    print("Errors found:", len(errors))

    for error in errors[:30]:
        print("ERROR:", error)
    if len(errors) > 30:
        print(f"... and {len(errors) - 30} more errors")

    if errors:
        sys.exit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
