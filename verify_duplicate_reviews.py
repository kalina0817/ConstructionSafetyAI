
from pathlib import Path
import json
from collections import Counter

project = Path(".")
report = project / "data/reports/near_duplicate_check"
review_file = report / "visual_review_results.json"

if not review_file.exists():
    raise FileNotFoundError(f"Review file not found: {review_file}")

data = json.loads(review_file.read_text(encoding="utf-8"))
pairs = data.get("pairs", [])

decisions = Counter()
missing_images = []
cross_split = Counter()

for pair in pairs:
    decision = pair.get("decision", "missing_decision")
    decisions[decision] += 1

    for key in ("image_1", "image_2"):
        image = pair.get(key, {})
        path = Path(image.get("path", ""))
        if not path.is_absolute():
            path = project / path
        if not path.is_file():
            missing_images.append((pair.get("pair"), str(path)))

    if pair.get("crosses_splits"):
        cross_split[decision] += 1

print("DUPLICATE REVIEW VERIFICATION")
print("=" * 34)
print(f"Review records: {len(pairs)}")
print("\nDecision counts:")
for decision, count in sorted(decisions.items()):
    print(f"  {decision}: {count}")

print("\nCross-split decisions:")
for decision, count in sorted(cross_split.items()):
    print(f"  {decision}: {count}")

print(f"\nImage paths checked: {len(pairs) * 2}")
print(f"Missing image paths: {len(missing_images)}")
for pair_number, path in missing_images:
    print(f"  Pair {pair_number}: {path}")

print("\nNo images, labels, or splits were modified.")
print("Training was not started.")
