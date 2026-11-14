from pathlib import Path
import shutil
import yaml

SOURCE = Path("data/processed/construction-toolset-deduplicated")
OUTPUT = Path("data/processed/construction-toolset-deduplicated-detection")
SPLITS = ("train", "valid", "test")


def convert_line(line, class_count):
    parts = line.split()

    if len(parts) < 2:
        raise ValueError("Annotation line is incomplete")

    try:
        class_value = float(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        raise ValueError("Annotation contains non-numeric values")

    if not class_value.is_integer():
        raise ValueError("Class ID must be an integer")

    class_id = int(class_value)

    if not 0 <= class_id < class_count:
        raise ValueError(f"Invalid class ID: {class_id}")

    if not all(0 <= value <= 1 for value in coords):
        raise ValueError("Coordinate outside the normalized 0–1 range")

    # Existing YOLO bounding box
    if len(coords) == 4:
        xc, yc, width, height = coords

        if width <= 0 or height <= 0:
            raise ValueError("Bounding box width and height must be positive")

        if xc - width / 2 < -1e-8 or xc + width / 2 > 1 + 1e-8:
            raise ValueError("Bounding box extends beyond image horizontally")

        if yc - height / 2 < -1e-8 or yc + height / 2 > 1 + 1e-8:
            raise ValueError("Bounding box extends beyond image vertically")

        values = [class_id, xc, yc, width, height]
        return " ".join(
            str(value) if index == 0 else f"{value:.8f}"
            for index, value in enumerate(values)
        ), "box"

    # YOLO segmentation polygon
    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError("Polygon must contain at least three coordinate pairs")

    xs = coords[0::2]
    ys = coords[1::2]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = xmax - xmin
    height = ymax - ymin

    if width <= 0 or height <= 0:
        raise ValueError("Polygon has zero width or height")

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2

    box = [class_id, xc, yc, width, height]

    return " ".join(
        str(value) if index == 0 else f"{value:.8f}"
        for index, value in enumerate(box)
    ), "polygon"

def main():
    yaml_path = SOURCE / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = config["names"]
    class_count = int(config["nc"])

    if len(names) != class_count:
        raise ValueError("Class count does not match class names")

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT}\n"
            "Nothing was changed. Use a new empty output folder."
        )

    totals = {"box": 0, "polygon": 0, "empty_files": 0}

    for split in SPLITS:
        src_images = SOURCE / split / "images"
        src_labels = SOURCE / split / "labels"
        dst_images = OUTPUT / split / "images"
        dst_labels = OUTPUT / split / "labels"

        if not src_images.is_dir() or not src_labels.is_dir():
            raise FileNotFoundError(f"Missing images or labels for {split}")

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        image_stems = {
            p.stem for p in src_images.iterdir()
            if p.is_file() and p.suffix.lower() in
            {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        }
        label_stems = {p.stem for p in src_labels.glob("*.txt")}

        if image_stems != label_stems:
            raise ValueError(f"Image/label mismatch in {split}")

        split_counts = {"box": 0, "polygon": 0, "empty_files": 0}

        for image in src_images.iterdir():
            if image.is_file() and image.stem in image_stems:
                shutil.copy2(image, dst_images / image.name)

        for label_file in sorted(src_labels.glob("*.txt")):
            raw = label_file.read_text(encoding="utf-8").splitlines()
            output_lines = []

            for line in raw:
                if not line.strip():
                    continue

                converted, kind = convert_line(line, class_count)

                class_id = int(float(line.split()[0]))
                if not 0 <= class_id < class_count:
                    raise ValueError(
                        f"Invalid class ID in {label_file}: {class_id}"
                    )

                output_lines.append(converted)
                split_counts[kind] += 1
                totals[kind] += 1

            if not output_lines:
                split_counts["empty_files"] += 1
                totals["empty_files"] += 1

            (dst_labels / label_file.name).write_text(
                "\n".join(output_lines) + ("\n" if output_lines else ""),
                encoding="utf-8",
            )

        print(f"\n{split}:")
        for key, value in split_counts.items():
            print(f"  {key}: {value}")

    output_config = dict(config)
    output_config["path"] = str(OUTPUT.resolve())
    output_config["train"] = "train/images"
    output_config["val"] = "valid/images"
    output_config["test"] = "test/images"

    (OUTPUT / "data.yaml").write_text(
        yaml.safe_dump(output_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("\nConversion completed.")
    print("Output:", OUTPUT.resolve())
    print("Original dataset was not modified.")
    print("Annotation totals:", totals)


if __name__ == "__main__":
    main()