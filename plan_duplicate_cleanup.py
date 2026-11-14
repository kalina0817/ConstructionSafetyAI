
from pathlib import Path
import json
from collections import defaultdict

base = Path("data/reports/near_duplicate_check")
data = json.loads(
    (base / "visual_review_results.json").read_text(encoding="utf-8")
)
output = base / "proposed_duplicate_cleanup_plan.txt"

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

pairs = [
    p for p in data["pairs"]
    if p.get("decision") == "exact_or_near_duplicate"
    and p.get("crosses_splits")
]

for p in pairs:
    union(p["image_1"]["path"], p["image_2"]["path"])

groups = defaultdict(dict)
for p in pairs:
    for key in ("image_1", "image_2"):
        image = p[key]
        groups[find(image["path"])][image["path"]] = image["split"]

lines = [
    "PROPOSED DUPLICATE CLEANUP PLAN — NO CHANGES MADE",
    "=" * 54,
    f"Flagged duplicate pairs: {len(pairs)}",
    f"Connected image groups: {len(groups)}",
    "",
    "Review groups containing images from multiple splits.",
    "Do not delete or move images until the plan is approved.",
]

for i, images in enumerate(
    sorted(groups.values(), key=lambda g: (-len(g), sorted(g)[0])), 1
):
    split_names = sorted(set(images.values()))
    lines += [
        "",
        f"GROUP {i}: {len(images)} images; splits: {', '.join(split_names)}"
    ]
    for path, split in sorted(images.items()):
        lines.append(f"  [{split}] {path}")

output.write_text("\n".join(lines), encoding="utf-8")

print(f"Plan saved: {output}")
print(f"Groups listed: {len(groups)}")
print("No images, labels, or dataset splits were modified.")
print("Training was not started.")
