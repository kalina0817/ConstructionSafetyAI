
from pathlib import Path
import json

project = Path(".")
report_dir = project / "data" / "reports" / "near_duplicate_check"

review_file = report_dir / "visual_review_results.json"
output_file = report_dir / "conservative_duplicate_plan.txt"

if not review_file.exists():
    raise FileNotFoundError(f"Review file not found: {review_file}")

data = json.loads(review_file.read_text(encoding="utf-8"))
pairs = data.get("pairs", [])

lines = [
    "CONSERVATIVE DUPLICATE PLAN — REPORT ONLY",
    "=" * 48,
    f"Reviewed pairs: {len(pairs)}",
    "",
    "No files were changed. No training was started.",
    "Pairs are listed individually; no connected groups are assumed.",
    "A duplicate decision is a candidate, not independent proof.",
    "Do not delete or move files based on this report alone.",
    "",
]

duplicate_count = 0
uncertain_count = 0
similar_count = 0

for item in pairs:
    decision = item.get("decision", "unknown")
    image1 = item.get("image_1", {})
    image2 = item.get("image_2", {})

    if decision == "exact_or_near_duplicate":
        duplicate_count += 1
        status = "CANDIDATE — CONFIRM BEFORE CLEANUP"
    elif decision == "similar_but_different":
        similar_count += 1
        status = "KEEP SEPARATE — SIMILAR PHOTOGRAPHS"
    else:
        uncertain_count += 1
        status = "UNCERTAIN — DO NOT AUTO-CLEAN"

    lines.extend([
        f"Pair {item.get('pair', '?')} | distance: {item.get('distance', 'unknown')}",
        f"Decision: {decision}",
        f"Status: {status}",
        f"Image 1 [{image1.get('split', '?')}]: {image1.get('path', 'missing path')}",
        f"Image 2 [{image2.get('split', '?')}]: {image2.get('path', 'missing path')}",
        "",
    ])

lines.extend([
    "SUMMARY",
    "-" * 48,
    f"Duplicate candidates requiring confirmation: {duplicate_count}",
    f"Similar-but-different pairs to keep separate: {similar_count}",
    f"Uncertain/unknown pairs: {uncertain_count}",
    "",
    "NEXT STEP: Independently verify candidates before creating a cleaned copy.",
    "Original dataset, labels, and splits remain unchanged.",
])

output_file.write_text("\n".join(lines), encoding="utf-8")

print(f"Report saved: {output_file}")
print(f"Pairs listed: {len(pairs)}")
print(f"Duplicate candidates: {duplicate_count}")
print(f"Similar-but-different pairs: {similar_count}")
print(f"Uncertain/unknown pairs: {uncertain_count}")
print("No dataset files were modified.")
print("Training was not started.")
