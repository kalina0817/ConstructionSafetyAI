
from pathlib import Path
import json
from collections import defaultdict

base = Path("data/reports/near_duplicate_check")
data = json.loads(
    (base / "visual_review_results.json").read_text(encoding="utf-8")
)

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
    "FINAL DUPLICATE AUDIT — PROPOSAL ONLY",
    "=" * 42,
    f"Candidate pairs: {len(pairs)}",
    f"Connected groups: {len(groups)}",
    "",
    "No files were changed.",
    "Suggested policy: keep confirmed duplicate photographs in one split.",
    "Do not automatically remove similar-but-different photographs.",
    "",
]

for i, images in enumerate(
    sorted(groups.values(), key=lambda g: (-len(g), sorted(g)[0])), 1
):
    splits = sorted(set(images.values()))
    action = (
        "REVIEW SPLIT ASSIGNMENT"
        if len(splits) > 1
        else "NO CROSS-SPLIT ACTION INDICATED"
    )
    lines += [
        f"Group {i}: {len(images)} images | {', '.join(splits)}",
        f"  Proposed status: {action}",
    ]

output = base / "final_duplicate_audit.txt"
output.write_text("\n".join(lines), encoding="utf-8")

print(f"Final audit saved: {output}")
print(f"Groups: {len(groups)}")
print("Images, labels, and splits were not modified.")
print("Training was not started.")
