
from pathlib import Path
import yaml
import math

ROOT = Path("data/raw")

DATASETS = {
    "construction-toolset": ["train", "valid", "test"],
    "construction-vehicle-detection": ["train", "valid"],
    "construction-ppe": ["train", "valid", "test"],
}

def get_class_count(dataset_path):
    config = dataset_path / "data.yaml"
    with config.open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)

    nc = data.get("nc")
    names = data.get("names")

    if not isinstance(nc, int) or not isinstance(names, (list, dict)):
        raise ValueError(f"Invalid class configuration: {config}")

    if len(names) != nc:
        raise ValueError(
            f"Class count mismatch in {config}: nc={nc}, names={len(names)}"
        )

    return nc

def audit_label_file(path, nc):
    issues = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    valid_objects = 0

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            issues.append((line_number, "Expected 5 values"))
            continue

        try:
            class_value = float(parts[0])
            box = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append((line_number, "Non-numeric value"))
            continue

        if not math.isfinite(class_value) or not class_value.is_integer():
            issues.append((line_number, "Class ID is not an integer"))
            continue

        class_id = int(class_value)
        if not 0 <= class_id < nc:
            issues.append((line_number, f"Class ID {class_id} outside 0..{nc-1}"))
            continue

        if not all(math.isfinite(value) for value in box):
            issues.append((line_number, "Non-finite box coordinate"))
            continue

        x, y, width, height = box
        if not (0 <= x <= 1 and 0 <= y <= 1):
            issues.append((line_number, "Center coordinate outside 0..1"))
            continue

        if not (0 < width <= 1 and 0 < height <= 1):
            issues.append((line_number, "Box width/height outside valid range"))
            continue

        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
            issues.append((line_number, "Box extends beyond image horizontally"))
            continue

        if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            issues.append((line_number, "Box extends beyond image vertically"))
            continue

        valid_objects += 1

    return valid_objects, issues, len(lines) == 0

def main():
    for dataset_name, splits in DATASETS.items():
        dataset_path = ROOT / dataset_name
        print(f"\n{'=' * 65}\nDATASET: {dataset_name}")

        try:
            nc = get_class_count(dataset_path)
        except Exception as exc:
            print(f"CONFIG ERROR: {exc}")
            continue

        print(f"Configured classes: {nc}")

        for split in splits:
            labels_dir = dataset_path / split / "labels"
            if not labels_dir.is_dir():
                print(f"{split}: labels folder not found; skipped")
                continue

            files = sorted(labels_dir.glob("*.txt"))
            total_objects = 0
            issue_count = 0
            empty_count = 0

            for path in files:
                try:
                    objects, issues, is_empty = audit_label_file(path, nc)
                except (OSError, UnicodeError) as exc:
                    issue_count += 1
                    print(f"READ ERROR: {path}: {exc}")
                    continue

                total_objects += objects
                issue_count += len(issues)

                if is_empty:
                    empty_count += 1

                for line_number, message in issues:
                    print(f"ISSUE: {path} line {line_number}: {message}")

            print(
                f"{split}: label files={len(files)}, "
                f"valid objects={total_objects}, "
                f"issues={issue_count}, empty files={empty_count}"
            )

if __name__ == "__main__":
    main()
