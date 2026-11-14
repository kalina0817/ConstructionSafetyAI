from pathlib import Path
from collections import defaultdict
import hashlib

dataset = Path("data/processed/construction-ppe-corrected")
splits = ["train", "valid", "test"]
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Group images by SHA-256 content hash.
hash_groups = defaultdict(list)

for split in splits:
    image_dir = dataset / split / "images"

    for image in sorted(image_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in extensions:
            continue

        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        hash_groups[digest].append((split, image.name))

# Report groups containing identical image files in different splits.
duplicates = []

for entries in hash_groups.values():
    split_names = {split for split, _ in entries}

    if len(split_names) > 1:
        duplicates.append(entries)

print("Dataset:", dataset)
print("Unique image files hashed:", sum(len(v) for v in hash_groups.values()))
print("Duplicate groups across splits:", len(duplicates))

for number, group in enumerate(duplicates, start=1):
    print(f"\nDuplicate group {number}:")
    for split, filename in group:
        print(f"  {split}: {filename}")

print("\nCheck complete. No files were changed.")