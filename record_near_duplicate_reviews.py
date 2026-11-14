
from pathlib import Path
import json
import re
from collections import Counter

report = Path(
    "data/reports/near_duplicate_check/near_duplicate_pairs.txt"
)
output = Path(
    "data/reports/near_duplicate_check/visual_review_results.json"
)

# Pairs you marked "Exact or near duplicate".
duplicate_pairs = {
    1, 6, 7, 8, 9, 10,
    11, 16, 17, 18, 20,
    21, 22, 23, 28, 29, 30,
    32, 33, 34, 35, 37,
    41, 42, 43, 46, 47, 49, 50,
    51, 52, 53, 54,
    68, 69,
    72, 77, 78, 79, 80,
    88,
}

if not report.exists():
    raise FileNotFoundError(f"Report not found: {report}")

text = report.read_text(encoding="utf-8")

# Extract distance and the two image paths for each candidate pair.
pattern = re.compile(
    r"Distance:\s*(\d+)\s*\n"
    r"\s*(train|valid|test):\s*(.+)\n"
    r"\s*(train|valid|test):\s*(.+)",
    re.IGNORECASE,
)
matches = list(pattern.finditer(text))

if len(matches) != 91:
    raise ValueError(
        f"Expected 91 pairs in the report, found {len(matches)}. "
        "No review file was written."
    )

results = []
cross_split_duplicates = []
cross_split_similar = []

for pair_number, match in enumerate(matches, start=1):
    distance, split1, path1, split2, path2 = match.groups()

    decision = (
        "exact_or_near_duplicate"
        if pair_number in duplicate_pairs
        else "similar_but_different"
    )

    item = {
        "pair": pair_number,
        "distance": int(distance),
        "image_1": {"split": split1.lower(), "path": path1.strip()},
        "image_2": {"split": split2.lower(), "path": path2.strip()},
        "decision": decision,
        "crosses_splits": split1.lower() != split2.lower(),
    }
    results.append(item)

    if item["crosses_splits"]:
        if decision == "exact_or_near_duplicate":
            cross_split_duplicates.append(pair_number)
        else:
            cross_split_similar.append(pair_number)

counts = Counter(item["decision"] for item in results)

review = {
    "status": "VISUAL_REVIEW_RECORDED",
    "source_report": str(report),
    "dataset_modified": False,
    "training_started": False,
    "pairs_reviewed": len(results),
    "exact_or_near_duplicate_pairs": counts["exact_or_near_duplicate"],
    "similar_but_different_pairs": counts["similar_but_different"],
    "cross_split_duplicate_pair_numbers": cross_split_duplicates,
    "cross_split_similar_pair_numbers": cross_split_similar,
    "pairs": results,
}

output.write_text(
    json.dumps(review, indent=2),
    encoding="utf-8",
)

print("Review file saved:", output)
print("Pairs recorded:", len(results))
print("Exact or near duplicates:", counts["exact_or_near_duplicate"])
print("Similar but different:", counts["similar_but_different"])
print("Duplicate pairs crossing splits:", len(cross_split_duplicates))
print("Similar pairs crossing splits:", len(cross_split_similar))
print("Cross-split duplicate pair numbers:", cross_split_duplicates)
print("No images or labels were changed.")
print("Training was not started.")
