
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path("data/raw/construction-ppe")
config = yaml.safe_load((ROOT / "data.yaml").read_text(encoding="utf-8"))
names = config["names"]

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
total_counts = Counter()

for split in ("train", "valid", "test"):
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in label_dir.glob("*.txt")}

    orphan_labels = sorted(label_stems - image_stems)
    images_without_labels = sorted(image_stems - label_stems)

    counts = Counter()
    for label in label_dir.glob("*.txt"):
        if label.stem not in image_stems:
            continue

        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id = int(float(line.split()[0]))
            counts[class_id] += 1
            total_counts[class_id] += 1

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Orphan label files:", len(orphan_labels))
    print("Images without labels:", len(images_without_labels))
    print("Annotations by class:")
    for class_id, name in enumerate(names):
        print(f"  {class_id}: {name}: {counts[class_id]}")

print("\nTOTAL ANNOTATIONS BY CLASS")
for class_id, name in enumerate(names):
    print(f"{class_id}: {name}: {total_counts[class_id]}")

print("\nAnalysis only. No files modified.")
