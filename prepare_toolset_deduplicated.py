
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import shutil
import yaml

SOURCE = Path(r"data\raw\construction-toolset")
DEST = Path(r"data\processed\construction-toolset-deduplicated")
SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}
SPLITS = ["train", "valid", "test"]

if not SOURCE.is_dir():
    raise SystemExit(f"Source dataset not found: {SOURCE}")

if DEST.exists() and any(DEST.iterdir()):
    raise SystemExit(
        f"Destination is not empty; nothing changed: {DEST}"
    )

config_path = SOURCE / "data.yaml"
if not config_path.exists():
    raise SystemExit("Source data.yaml not found.")

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
names = config.get("names")

if not isinstance(names, list) or len(names) != config.get("nc"):
    raise SystemExit("Class configuration is inconsistent; nothing copied.")

groups = defaultdict(list)
missing_labels = []
source_images = 0

# Inventory and validate before copying anything.
for split in SPLITS:
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"Missing images or labels folder for {split}.")

    for image in image_dir.iterdir():
        if not image.is_file():
            continue

        source_images += 1
        label = label_dir / f"{image.stem}.txt"

        if not label.is_file():
            missing_labels.append(str(label))
            continue

        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        groups[digest].append({
            "split": split,
            "image": image.name,
            "label": label.name,
        })

if missing_labels:
    print("Missing label files:")
    for item in missing_labels[:20]:
        print(item)
    raise SystemExit("Preflight failed; nothing copied.")

# Deterministically select one copy per image hash.
selected = []
duplicate_details = []

for digest, records in sorted(groups.items()):
    ordered = sorted(
        records,
        key=lambda r: (SPLIT_PRIORITY[r["split"]], r["image"].lower())
    )
    keep = ordered[0]
    selected.append(keep)

    if len(records) > 1:
        label_hashes = set()
        for record in records:
            label_path = (
                SOURCE / record["split"] / "labels" / record["label"]
            )
            label_hashes.add(
                hashlib.sha256(label_path.read_bytes()).hexdigest()
            )

        duplicate_details.append({
            "sha256": digest,
            "selected": keep,
            "selected_by_priority": keep["split"],
            "labels_identical": len(label_hashes) == 1,
            "source_copies": records,
        })

# Create a new destination; never overwrite a populated folder.
DEST.mkdir(parents=True, exist_ok=True)
split_counts = Counter()

for record in selected:
    split = record["split"]
    source_image = SOURCE / split / "images" / record["image"]
    source_label = SOURCE / split / "labels" / record["label"]

    target_images = DEST / split / "images"
    target_labels = DEST / split / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_image, target_images / record["image"])
    shutil.copy2(source_label, target_labels / record["label"])
    split_counts[split] += 1

new_config = {
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": config["nc"],
    "names": names,
}
(DEST / "data.yaml").write_text(
    yaml.safe_dump(new_config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

audit = {
    "source": str(SOURCE),
    "destination": str(DEST),
    "source_images": source_images,
    "unique_images_retained": len(selected),
    "duplicate_groups": len(duplicate_details),
    "duplicate_copies_excluded": source_images - len(selected),
    "duplicate_groups_with_different_label_files": sum(
        not item["labels_identical"] for item in duplicate_details
    ),
    "split_counts": dict(split_counts),
    "selection_priority": ["test", "valid", "train"],
    "class_count": config["nc"],
    "class_names": names,
    "duplicate_details": duplicate_details,
    "note": (
        "Images and labels are copied unchanged. Mixed bounding-box and "
        "polygon annotation formats are preserved and require separate "
        "preparation before standard YOLO detection training."
    ),
}
(DEST / "audit_report.json").write_text(
    json.dumps(audit, indent=2),
    encoding="utf-8",
)

print("\nDEDUPLICATED TOOL SET CREATED")
print("Destination:", DEST.resolve())
print("Source images checked:", source_images)
print("Unique images retained:", len(selected))
print("Duplicate groups:", len(duplicate_details))
print("Duplicate copies excluded:", source_images - len(selected))
print(
    "Duplicate groups with differing label files:",
    audit["duplicate_groups_with_different_label_files"],
)
for split in SPLITS:
    print(f"{split}: {split_counts[split]} images")
print("Classes:", config["nc"])
print("Audit report:", DEST / "audit_report.json")
print("Original dataset was not modified.")
print("No model training was started.")
