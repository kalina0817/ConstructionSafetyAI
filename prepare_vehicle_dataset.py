
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import random
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "raw" / "construction-vehicle-detection"
DEST = ROOT / "data" / "processed" / "construction-vehicle-detection-deduplicated"

CLASSES = [
    "Boringmachine", "Bulldozer", "Dump Truck", "Excavator",
    "ForkLift", "Grader", "Loader", "Mixer Truck",
    "Mobile Crane", "Roller",
]

SPLITS = {"train": 0.70, "valid": 0.15, "test": 0.15}
SOURCE_SPLITS = ("train", "valid")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEED = 42


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label(path):
    present = set()
    boxes = Counter()
    invalid = []

    lines = path.read_text(encoding="utf-8-sig").splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        parts = line.split()

        try:
            if len(parts) != 5:
                raise ValueError("expected 5 values")

            class_value = float(parts[0])
            coords = [float(value) for value in parts[1:]]

            if not math.isfinite(class_value) or not class_value.is_integer():
                raise ValueError("class ID must be an integer")

            class_id = int(class_value)

            if not 0 <= class_id < len(CLASSES):
                raise ValueError("class ID outside configured range")

            if not all(math.isfinite(value) for value in coords):
                raise ValueError("non-finite coordinate")

            x, y, width, height = coords

            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("center coordinate outside [0, 1]")

            if not (0 < width <= 1 and 0 < height <= 1):
                raise ValueError("box width/height outside (0, 1]")

            if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
                raise ValueError("box extends beyond image horizontally")

            if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
                raise ValueError("box extends beyond image vertically")

            present.add(class_id)
            boxes[class_id] += 1

        except (ValueError, OverflowError) as error:
            invalid.append(f"line {line_number}: {error}")

    return present, boxes, invalid


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    # An existing empty destination is allowed. Never overwrite contents.
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(
            "Destination is not empty. Nothing was changed:\n"
            f"{DEST}"
        )

    rng = random.Random(SEED)
    all_records = []
    problems = []
    duplicate_groups = {}

    # Read both source splits. Never modify the source dataset.
    for split in SOURCE_SPLITS:
        image_dir = SOURCE / split / "images"
        label_dir = SOURCE / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing images or labels folder for source split: {split}"
            )

        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )

        labels_by_stem = {
            path.stem: path
            for path in label_dir.glob("*.txt")
            if path.is_file()
        }
        image_stems = {path.stem for path in images}

        for image in images:
            label = labels_by_stem.get(image.stem)

            if label is None:
                problems.append(f"Missing label: {image}")
                continue

            try:
                present, boxes, invalid = read_label(label)
                digest = sha256(image)
            except (OSError, UnicodeError) as error:
                problems.append(f"Could not read {image} or {label}: {error}")
                continue

            for issue in invalid:
                problems.append(f"Invalid annotation in {label}: {issue}")

            record = {
                "image": image,
                "label": label,
                "hash": digest,
                "classes": present,
                "boxes": boxes,
                "source_split": split,
            }

            all_records.append(record)
            duplicate_groups.setdefault(digest, []).append(record)

        for stem, label in labels_by_stem.items():
            if stem not in image_stems:
                problems.append(f"Label without image: {label}")

    if problems:
        print("ERRORS FOUND. No dataset was created.")
        for problem in problems[:100]:
            print(problem)
        print(f"Total problems: {len(problems)}")
        return

    # Keep one deterministic representative of each exact image.
    # Source splits and filenames are sorted, so the choice is reproducible.
    unique_records = []
    duplicate_report = []

    for digest, group in sorted(duplicate_groups.items()):
        representative = group[0]
        unique_records.append(representative)

        if len(group) > 1:
            label_hashes = {
                hashlib.sha256(record["label"].read_bytes()).hexdigest()
                for record in group
            }

            duplicate_report.append({
                "sha256": digest,
                "retained_image": representative["image"].name,
                "retained_label": representative["label"].name,
                "retained_source_split": representative["source_split"],
                "excluded_copies": [
                    {
                        "image": record["image"].name,
                        "label": record["label"].name,
                        "source_split": record["source_split"],
                    }
                    for record in group[1:]
                ],
                "copies_in_source_group": len(group),
                "labels_identical": len(label_hashes) == 1,
            })

    # Assign unique images to splits. Each exact image can appear only once.
    class_image_totals = Counter()
    for record in unique_records:
        class_image_totals.update(record["classes"])

    targets = {
        split: len(unique_records) * fraction
        for split, fraction in SPLITS.items()
    }
    class_targets = {
        split: {
            class_id: class_image_totals[class_id] * fraction
            for class_id in range(len(CLASSES))
        }
        for split, fraction in SPLITS.items()
    }

    rng.shuffle(unique_records)

    def rarity(record):
        return min(
            (class_image_totals[class_id] for class_id in record["classes"]),
            default=float("inf"),
        )

    unique_records.sort(key=rarity)

    assigned = {split: [] for split in SPLITS}
    split_sizes = Counter()
    split_class_counts = {
        split: Counter() for split in SPLITS
    }

    for record in unique_records:
        best_split = None
        best_score = float("-inf")

        for split in SPLITS:
            size_deficit = targets[split] - split_sizes[split]
            score = size_deficit / max(targets[split], 1)

            for class_id in record["classes"]:
                deficit = (
                    class_targets[split][class_id]
                    - split_class_counts[split][class_id]
                )
                score += deficit / max(class_targets[split][class_id], 1)

            if score > best_score:
                best_score = score
                best_split = split

        assigned[best_split].append(record)
        split_sizes[best_split] += 1
        split_class_counts[best_split].update(record["classes"])

    report = {
        "source": str(SOURCE),
        "destination": str(DEST),
        "seed": SEED,
        "classes": CLASSES,
        "requested_split_ratios": SPLITS,
        "source_splits_read": list(SOURCE_SPLITS),
        "source_images": len(all_records),
        "unique_images_retained": len(unique_records),
        "duplicate_groups": len(duplicate_report),
        "duplicate_copies_excluded": (
            len(all_records) - len(unique_records)
        ),
        "exact_duplicate_groups": duplicate_report,
        "splits": {},
    }

    # The destination must be empty before creating the dataset.
    DEST.mkdir(parents=True, exist_ok=True)

    try:
        for split, records in assigned.items():
            image_dir = DEST / split / "images"
            label_dir = DEST / split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            box_counts = Counter()
            image_counts = Counter()

            for record in records:
                shutil.copy2(
                    record["image"], image_dir / record["image"].name
                )
                shutil.copy2(
                    record["label"], label_dir / record["label"].name
                )
                box_counts.update(record["boxes"])
                image_counts.update(record["classes"])

            report["splits"][split] = {
                "images": len(records),
                "label_files": len(records),
                "boxes_per_class": {
                    CLASSES[class_id]: box_counts[class_id]
                    for class_id in range(len(CLASSES))
                },
                "images_per_class": {
                    CLASSES[class_id]: image_counts[class_id]
                    for class_id in range(len(CLASSES))
                },
            }

        yaml_lines = [
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            f"nc: {len(CLASSES)}",
            "names: " + json.dumps(CLASSES, ensure_ascii=False),
            "",
        ]

        (DEST / "data.yaml").write_text(
            "\n".join(yaml_lines), encoding="utf-8"
        )
        (DEST / "audit_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    except Exception:
        print(
            "\nDataset creation failed. The raw source was not modified.\n"
            "The destination may contain partial output; inspect it before "
            "trying again:\n"
            f"{DEST}"
        )
        raise

    print("\nDEDUPLICATED DATASET CREATED")
    print(f"Destination: {DEST}")
    print(f"Source images checked: {len(all_records)}")
    print(f"Unique images retained: {len(unique_records)}")
    print(f"Exact duplicate groups: {len(duplicate_report)}")
    print(f"Duplicate copies excluded: {len(all_records) - len(unique_records)}")

    for split, stats in report["splits"].items():
        print(f"\n{split.upper()}: {stats['images']} images")

    print(f"\nDataset configuration: {DEST / 'data.yaml'}")
    print(f"Audit report: {DEST / 'audit_report.json'}")
    print("No model training was started.")
    print("Review the audit report before training.")


if __name__ == "__main__":
    main()
