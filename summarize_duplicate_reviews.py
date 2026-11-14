
from pathlib import Path
import json
import re
from collections import defaultdict

base = Path("data/reports/near_duplicate_check")
review_file = base / "visual_review_results.json"
pixel_file = base / "pixel_difference_all_pairs.txt"
output_file = base / "duplicate_review_summary.txt"

reviews = json.loads(review_file.read_text(encoding="utf-8"))
pixel_text = pixel_file.read_text(encoding="utf-8")

# Read pixel-difference measurements.
pixel_results = {}
for line in pixel_text.splitlines():
    match = re.search(
        r"Pair\s+(\d+):\s*mean difference\s*=\s*([\d.]+)/255;"
        r".*?splits\s*=\s*(.+)",
        line
    )
    if match:
        pixel_results[int(match.group(1))] = {
            "mean_difference": float(match.group(2)),
            "splits": match.group(3).strip()
        }

pairs = reviews.get("pairs", [])
duplicate_decisions = [
    p for p in pairs
    if p.get("decision") == "exact_or_near_duplicate"
    and p.get("crosses_splits")
]
similar_decisions = [
    p for p in pairs
    if p.get("decision") == "similar_but_different"
    and p.get("crosses_splits")
]

# Group confirmed visual duplicate decisions by image path.
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for p in duplicate_decisions:
    union(p["image_1"]["path"], p["image_2"]["path"])

groups = defaultdict(set)
for p in duplicate_decisions:
    for key in ("image_1", "image_2"):
        path = p[key]["path"]
        groups[find(path)].add(path)

lines = [
    "CONSTRUCTION PPE NEAR-DUPLICATE REVIEW SUMMARY",
    "=" * 48,
    f"Candidate pairs reviewed: {reviews.get('pairs_reviewed', 0)}",
    f"Visually marked exact/near duplicate pairs: {len(duplicate_decisions)}",
    f"Visually marked similar-but-different pairs: {len(similar_decisions)}",
    f"Pixel-difference measurements available: {len(pixel_results)}",
    f"Connected groups among duplicate-marked pairs: {len(groups)}",
    "",
    "DUPLICATE-MARKED CROSS-SPLIT PAIRS",
    "-" * 35,
]

for p in duplicate_decisions:
    n = p["pair"]
    measurement = pixel_results.get(n)
    diff = (
        f"{measurement['mean_difference']:.2f}/255"
        if measurement else "not measured"
    )
    lines.append(
        f"Pair {n}: {p['image_1']['split']} vs "
        f"{p['image_2']['split']}; "
        f"perceptual hash distance={p.get('distance', 'unknown')}; "
        f"mean pixel difference={diff}"
    )
    lines.append(f"  Image 1: {p['image_1']['path']}")
    lines.append(f"  Image 2: {p['image_2']['path']}")

lines += [
    "",
    "IMPORTANT",
    "-" * 9,
    "This report summarizes existing review decisions; it does not delete or move files.",
    "Mean pixel difference is not a duplicate verdict by itself.",
    "Do not remove images automatically based only on these measurements.",
    "Dataset splits have not been changed by this script.",
    "Training has not been started by this script.",
]

output_file.write_text("\n".join(lines), encoding="utf-8")

print(f"Summary saved: {output_file}")
print(f"Duplicate-marked cross-split pairs: {len(duplicate_decisions)}")
print(f"Similar-but-different cross-split pairs: {len(similar_decisions)}")
print(f"Pixel measurements matched: {len(pixel_results)}")
print("No images, labels, or dataset splits were modified.")
print("Training was not started.")
