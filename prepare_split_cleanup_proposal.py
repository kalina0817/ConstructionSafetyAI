
from pathlib import Path
import json
from collections import defaultdict

report_dir = Path("data/reports/near_duplicate_check")
review_path = report_dir / "visual_review_results.json"
output_path = report_dir / "split_cleanup_proposal.txt"

data = json.loads(review_path.read_text(encoding="utf-8"))

# Build groups only from pairs explicitly marked as duplicates.
parent = {}

def find(item):
    parent.setdefault(item, item)
    if parent[item] != item:
        parent[item] = find(parent[item])
    return parent[item]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

confirmed_candidates = []
similar_pairs = []

for pair in data.get("pairs", []):
    decision = pair.get("decision")
    a = pair.get("image_1", {})
    b = pair.get("image_2", {})

    if decision == "similar_but_different":
        similar_pairs.append(pair)
        continue

    if decision != "exact_or_near_duplicate":
        continue

    if not pair.get("crosses_splits"):
        continue

    confirmed_candidates.append(pair)
    union(a["path"], b["path"])

groups = defaultdict(dict)
for pair in confirmed_candidates:
    for key in ("image_1", "image_2"):
        image = pair[key]
        groups[find(image["path"])][image["path"]] = image["split"]

lines = [
    "SPLIT CLEANUP PROPOSAL — NO DATASET CHANGES",
    "=" * 48,
    f"Cross-split duplicate-candidate pairs: {len(confirmed_candidates)}",
    f"Similar-but-different pairs excluded from cleanup: {len(similar_pairs)}",
    f"Connected candidate groups: {len(groups)}",
    "",
    "IMPORTANT:",
    "- This report does not authorize deletion, movement, or reassignment.",
    "- Visual review decisions are not independent proof of duplication.",
    "- Confirm each group before changing split assignments.",
    "- Keep related duplicate images in only one split.",
    "- Preserve a separate test set for final evaluation.",
    "",
]

for number, images in enumerate(
    sorted(groups.values(), key=lambda g: (-len(g), sorted(g)[0])), 1
):
    splits = sorted(set(images.values()))
    lines.append(
        f"Group {number}: {len(images)} images | splits: {', '.join(splits)}"
    )
    for path, split in sorted(images.items()):
        lines.append(f"  [{split}] {path}")
    lines.append("  Action: MANUAL CONFIRMATION REQUIRED")
    lines.append("")

lines += [
    "No images, labels, or dataset splits were modified.",
    "Training was not started.",
]

output_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Proposal saved: {output_path}")
print(f"Candidate pairs: {len(confirmed_candidates)}")
print(f"Similar pairs excluded: {len(similar_pairs)}")
print(f"Groups: {len(groups)}")
print("No dataset files were modified.")
print("Training was not started.")
