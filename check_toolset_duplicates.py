
import json
from collections import Counter
from pathlib import Path

report_path = Path(r"data\reports\toolset_duplicate_audit.json")
report = json.loads(report_path.read_text())

pairs = Counter()

for group in report["groups"]:
    splits = sorted({item["split"] for item in group})
    if len(splits) > 1:
        for i, first in enumerate(splits):
            for second in splits[i + 1:]:
                pairs[(first, second)] += 1

print("Cross-split duplicate group counts:")
for (first, second), count in sorted(pairs.items()):
    print(f"{first} <-> {second}: {count}")

print("\nExamples of cross-split groups:")
shown = 0
for group in report["groups"]:
    if len({item["split"] for item in group}) > 1:
        print(json.dumps(group, indent=2))
        shown += 1
        if shown == 5:
            break
