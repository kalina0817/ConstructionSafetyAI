
from pathlib import Path
import json
from collections import defaultdict

base = Path("data/reports/near_duplicate_check")
review_file = base / "visual_review_results.json"
output_file = base / "cross_split_duplicate_images.txt"

data = json.loads(review_file.read_text(encoding="utf-8"))

images_by_split = defaultdict(set)
duplicate_pairs = [
    p for p in data["pairs"]
    if p.get("decision") == "exact_or_near_duplicate"
    and p.get("crosses_splits")
]

for pair in duplicate_pairs:
    for key in ("image_1", "image_2"):
        image = pair[key]
        images_by_split[image["split"]].add(image["path"])

lines = [
    "UNIQUE IMAGES IN DUPLICATE-MARKED CROSS-SPLIT PAIRS",
    "=" * 52,
    f"Pairs: {len(duplicate_pairs)}",
]

for split in ("train", "valid", "test"):
    paths = sorted(images_by_split[split])
    lines += ["", f"{split.upper()}: {len(paths)} unique images"]
    lines.extend(paths)

output_file.write_text("\n".join(lines), encoding="utf-8")

print(f"Report saved: {output_file}")
for split in ("train", "valid", "test"):
    print(f"{split}: {len(images_by_split[split])} unique images")
print("No images, labels, or dataset splits were modified.")
print("Training was not started.")
