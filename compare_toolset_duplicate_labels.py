
from pathlib import Path
import hashlib
import json

ROOT = Path(r"data\raw\construction-toolset")
INPUT_REPORT = Path(r"data\reports\toolset_duplicate_audit.json")
OUTPUT_REPORT = Path(r"data\reports\toolset_duplicate_label_comparison.json")


def read_label(path):
    if not path.exists():
        return None, None

    raw = path.read_text(encoding="utf-8")
    parsed = []

    try:
        for line in raw.splitlines():
            parts = line.split()
            if not parts:
                continue

            class_id = int(parts[0])
            coordinates = tuple(round(float(v), 8) for v in parts[1:])
            parsed.append((class_id, coordinates))

        return raw, sorted(parsed)
    except (ValueError, IndexError):
        return raw, "INVALID"


report = json.loads(INPUT_REPORT.read_text(encoding="utf-8"))
results = []

for group in report["groups"]:
    if len(group) < 2:
        continue

    reference = group[0]
    reference_path = (
        ROOT / reference["split"] / "labels" / reference["label"]
    )
    reference_raw, reference_parsed = read_label(reference_path)

    comparisons = []

    for item in group[1:]:
        label_path = ROOT / item["split"] / "labels" / item["label"]
        raw, parsed = read_label(label_path)

        comparisons.append({
            "split": item["split"],
            "image": item["image"],
            "label": item["label"],
            "label_exists": raw is not None,
            "exact_text_match": (
                raw == reference_raw
                if raw is not None and reference_raw is not None
                else None
            ),
            "annotation_match": (
                parsed == reference_parsed
                if parsed is not None and reference_parsed is not None
                else None
            ),
        })

    results.append({
        "reference": reference,
        "reference_label_exists": reference_raw is not None,
        "comparisons": comparisons,
    })

identical = 0
different = 0
missing = 0
invalid = 0

for result in results:
    if not result["reference_label_exists"]:
        missing += 1
        continue

    for item in result["comparisons"]:
        if not item["label_exists"]:
            missing += 1
        elif item["annotation_match"] == "INVALID":
            invalid += 1
        elif item["annotation_match"] is True:
            identical += 1
        else:
            different += 1

OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("Duplicate groups compared:", len(results))
print("Matching annotations:", identical)
print("Different annotations:", different)
print("Missing label files:", missing)
print("Invalid annotation files:", invalid)
print("Detailed report:", OUTPUT_REPORT)

print("\nExamples with different annotations:")
shown = 0
for result in results:
    for item in result["comparisons"]:
        if item["annotation_match"] is False:
            print(
                result["reference"]["image"],
                "<->",
                item["image"],
            )
            shown += 1
            if shown >= 5:
                break
    if shown >= 5:
        break
