from pathlib import Path
from collections import defaultdict
from PIL import Image
import imagehash

dataset = Path("data/processed/construction-ppe-corrected")
splits = ["train", "valid", "test"]
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Calculate perceptual hashes for every image.
records = []

for split in splits:
    image_dir = dataset / split / "images"

    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        try:
            with Image.open(path) as image:
                phash = imagehash.phash(image.convert("RGB"))

            records.append({
                "split": split,
                "name": path.name,
                "path": str(path),
                "hash": phash,
            })
        except Exception as exc:
            print(f"Could not read {path}: {exc}")

print("Images hashed:", len(records))

# Compare images from different splits.
# Hamming distance <= 6 flags possible near-duplicates.
pairs = []

for i, first in enumerate(records):
    for second in records[i + 1:]:
        if first["split"] == second["split"]:
            continue

        distance = first["hash"] - second["hash"]

        if distance <= 6:
            pairs.append((distance, first, second))

pairs.sort(key=lambda item: item[0])

report_dir = Path("data/reports/near_duplicate_check")
report_dir.mkdir(parents=True, exist_ok=True)
report_file = report_dir / "near_duplicate_pairs.txt"

with report_file.open("w", encoding="utf-8") as f:
    f.write(f"Images hashed: {len(records)}\n")
    f.write(f"Candidate pairs (distance <= 6): {len(pairs)}\n\n")

    for distance, first, second in pairs:
        f.write(f"Distance: {distance}\n")
        f.write(f"  {first['split']}: {first['path']}\n")
        f.write(f"  {second['split']}: {second['path']}\n\n")

print("Candidate near-duplicate pairs:", len(pairs))
print("Report saved to:", report_file)
print("No images or labels were modified.")