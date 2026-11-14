import hashlib
import math
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_KEYS = {
    "train": "train",
    "val": "val",
    "test": "test",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_count(config, data_path):
    count = config.get("nc")
    names = config.get("names")
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"Dataset YAML needs a positive integer nc: {data_path}")
    if isinstance(names, list):
        names_by_id = dict(enumerate(names))
    elif isinstance(names, dict):
        names_by_id = {int(key): value for key, value in names.items()}
    else:
        raise ValueError(f"Dataset YAML needs a names list or mapping: {data_path}")
    if len(names_by_id) != count or set(names_by_id) != set(range(count)):
        raise ValueError(
            f"Class names do not match nc={count} in {data_path}."
        )
    if any(not isinstance(value, str) or not value.strip()
           for value in names_by_id.values()):
        raise ValueError(f"Dataset class names must be non-empty strings.")
    if len({value.strip().lower() for value in names_by_id.values()}) != count:
        raise ValueError(f"Dataset class names contain duplicates: {data_path}")
    return names_by_id


def _resolve_dataset_root(config, yaml_path):
    configured_root = config.get("path")
    if configured_root is None:
        return yaml_path.parent.resolve()
    root = Path(configured_root).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def _split_path(config, split_key, dataset_root, yaml_path):
    value = config.get(split_key)
    if not value:
        return None
    if isinstance(value, list):
        paths = []
        for item in value:
            path = Path(item)
            if not path.is_absolute():
                path = dataset_root / path
            paths.append(path.resolve())
        return paths
    if not isinstance(value, str):
        raise ValueError(
            f"Unsupported {split_key} path value in {yaml_path}: {value!r}"
        )
    path = Path(value)
    if not path.is_absolute():
        path = dataset_root / path
    return [path.resolve()]


def _label_errors(label_path, class_count):
    errors = []
    annotation_count = 0
    class_counts = {}
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        return 0, {}, [f"Cannot read label {label_path}: {error}"]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("Expected class_id x_center y_center width height")
            values = [float(value) for value in parts]
            class_value, center_x, center_y, width, height = values
            if not all(math.isfinite(value) for value in values):
                raise ValueError("All annotation values must be finite")
            if not class_value.is_integer():
                raise ValueError("Class ID must be an integer")
            class_id = int(class_value)
            if not 0 <= class_id < class_count:
                raise ValueError(f"Class ID {class_id} outside [0, {class_count})")
            if not (
                0 <= center_x <= 1
                and 0 <= center_y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                raise ValueError("Coordinates must be normalized with positive size")
            if (
                center_x - width / 2 < -1e-6
                or center_x + width / 2 > 1 + 1e-6
                or center_y - height / 2 < -1e-6
                or center_y + height / 2 > 1 + 1e-6
            ):
                raise ValueError("Bounding box extends beyond image")
            annotation_count += 1
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        except (ValueError, OverflowError) as error:
            errors.append(
                f"{label_path} line {line_number}: {error} | {line}"
            )
    return annotation_count, class_counts, errors


def audit_dataset(data_yaml):
    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_yaml}")
    names_by_id = _class_count(config, data_yaml)
    dataset_root = _resolve_dataset_root(config, data_yaml)
    errors = []
    splits = {}

    for split_name, split_key in SPLIT_KEYS.items():
        image_paths = _split_path(
            config,
            split_key,
            dataset_root,
            data_yaml,
        )
        if image_paths is None:
            errors.append(f"Dataset YAML has no {split_key!r} split.")
            continue

        images = []
        for image_dir in image_paths:
            if not image_dir.is_dir():
                errors.append(f"{split_name} image directory not found: {image_dir}")
                continue
            images.extend(
                path for path in image_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        images.sort()

        label_dirs = {
            image_dir.parent / "labels"
            for image_dir in image_paths
        }
        label_files = {}
        for label_dir in label_dirs:
            if label_dir.is_dir():
                for path in label_dir.rglob("*.txt"):
                    if path.is_file():
                        if path.stem in label_files:
                            errors.append(
                                f"Duplicate label stem {path.stem!r} in "
                                f"{split_name}."
                            )
                        label_files[path.stem] = path

        image_by_stem = {}
        for image in images:
            if image.stem in image_by_stem:
                errors.append(
                    f"Duplicate image stem {image.stem!r} in {split_name}."
                )
            image_by_stem[image.stem] = image

        paired_stems = image_by_stem.keys() & label_files.keys()
        missing_labels = sorted(image_by_stem.keys() - label_files.keys())
        orphan_labels = sorted(label_files.keys() - image_by_stem.keys())
        if orphan_labels:
            errors.extend(
                f"{split_name} label has no image: {label_files[stem]}"
                for stem in orphan_labels[:100]
            )

        empty_label_files = 0
        annotation_count = 0
        split_class_counts = {class_id: 0 for class_id in names_by_id}
        for stem in paired_stems:
            label_path = label_files[stem]
            count, class_counts, label_errors = _label_errors(
                label_path,
                len(names_by_id),
            )
            errors.extend(label_errors[:100])
            if label_path.stat().st_size == 0:
                empty_label_files += 1
            annotation_count += count
            for class_id, class_count in class_counts.items():
                split_class_counts[class_id] += class_count

        unreadable_images = []
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as error:
                unreadable_images.append(f"{image_path}: {error}")
        errors.extend(unreadable_images[:100])

        splits[split_name] = {
            "image_count": len(images),
            "label_file_count": len(label_files),
            "paired_image_label_count": len(paired_stems),
            "images_without_label_file": len(missing_labels),
            "unpaired_image_examples": [
                image_by_stem[stem].name for stem in missing_labels[:20]
            ],
            "orphan_label_file_count": len(orphan_labels),
            "empty_label_file_count": empty_label_files,
            "annotation_count": annotation_count,
            "annotations_by_class": {
                names_by_id[class_id]: count
                for class_id, count in split_class_counts.items()
            },
            "unreadable_image_count": len(unreadable_images),
        }

    return {
        "schema_version": "1.0",
        "dataset_yaml": str(data_yaml),
        "dataset_yaml_sha256": _sha256(data_yaml),
        "dataset_root": str(dataset_root),
        "class_count": len(names_by_id),
        "class_names": names_by_id,
        "splits": splits,
        "errors": errors[:500],
        "error_count": len(errors),
        "interpretation_note": (
            "An image without a label file may be an intended negative or an "
            "unlabeled example. This structural audit cannot determine which; "
            "resolve and document the semantics before training or evaluation."
        ),
    }
