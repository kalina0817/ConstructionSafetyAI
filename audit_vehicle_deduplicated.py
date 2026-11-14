from pathlib import Path
import hashlib
import math
import yaml

ROOT = Path("data/processed/construction-vehicle-detection-deduplicated")
REPORT_DIR = Path("data/reports")
SPLITS = {"train": "train", "valid": "valid", "test": "test"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_yaml():
    config = ROOT / "data.yaml"
    with config.open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)

    nc = data.get("nc")
    names = data.get("names")
    if not isinstance(nc, int) or not isinstance(names, (list, dict)):
        raise ValueError(f"Invalid class configuration: {config}")
    if len(names) != nc:
        raise ValueError(f"Class count mismatch: nc={nc}, names={len(names)}")
    return nc, names

def main():
    nc, names = read_yaml()
    print(f"Dataset: {ROOT}")
    print(f"Classes: {nc}")
    print(f"Names: {names}")

    report = []
    all_hashes = {}
    total_images = total_labels = total_boxes = total_empty = total_issues = 0

    for split in SPLITS:
        image_dir = ROOT / split / "images"
        label_dir = ROOT / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images or labels folder for split: {split}")

        images = {p.stem: p for p in image_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
        labels = {p.stem: p for p in label_dir.glob("*.txt")}

        missing_labels = sorted(set(images) - set(labels))
        orphan_labels = sorted(set(labels) - set(images))
        split_boxes = split_empty = split_issues = 0

        for stem, image_path in images.items():
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            all_hashes.setdefault(digest, []).append((split, image_path.name))

        for stem, label_path in labels.items():
            lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            if not any(line.strip() for line in lines):
                split_empty += 1
                continue

            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                parts = line.split()
                issue = None

                if len(parts) != 5:
                    issue = "expected 5 values"
                else:
                    try:
                        values = [float(x) for x in parts]
                    except ValueError:
                        values = []
                        issue = "non-numeric value"

                    if values and issue is None:
                        cls, x, y, w, h = values
                        if not all(math.isfinite(v) for v in values):
                            issue = "non-finite value"
                        elif not cls.is_integer() or not 0 <= int(cls) < nc:
                            issue = "invalid class ID"
                        elif not (0 <= x <= 1 and 0 <= y <= 1 and
                                  0 < w <= 1 and 0 < h <= 1):
                            issue = "invalid normalized box"
                        elif x - w/2 < -1e-6 or x + w/2 > 1 + 1e-6:
                            issue = "box extends beyond image horizontally"
                        elif y - h/2 < -1e-6 or y + h/2 > 1 + 1e-6:
                            issue = "box extends beyond image vertically"

                if issue:
                    split_issues += 1
                    report.append(f"ISSUE,{split},{label_path.name},{line_no},{issue}")
                else:
                    split_boxes += 1

        report.extend(f"MISSING_LABEL,{split},{stem}" for stem in missing_labels)
        report.extend(f"ORPHAN_LABEL,{split},{stem}" for stem in orphan_labels)

        total_images += len(images)
        total_labels += len(labels)
        total_boxes += split_boxes
        total_empty += split_empty
        total_issues += split_issues

        print(f"\n{split.upper()}: images={len(images)}, labels={len(labels)}, "
              f"missing_labels={len(missing_labels)}, orphan_labels={len(orphan_labels)}, "
              f"valid_boxes={split_boxes}, empty_labels={split_empty}, "
              f"annotation_issues={split_issues}")

    duplicate_groups = [items for items in all_hashes.values() if len(items) > 1]
    print(f"\nTOTAL: images={total_images}, label_files={total_labels}, "
          f"valid_boxes={total_boxes}, empty_labels={total_empty}, "
          f"annotation_issues={total_issues}")
    print(f"Exact duplicate image groups across splits: {len(duplicate_groups)}")

    for group in duplicate_groups:
        report.append("DUPLICATE," + repr(group))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "vehicle_deduplicated_final_audit_20261027.txt"
    if output.exists():
        raise FileExistsError(f"Report already exists; refusing to overwrite: {output}")

    summary = [
        "VEHICLE DEDUPLICATED DATASET AUDIT",
        f"Dataset: {ROOT}",
        f"Images: {total_images}",
        f"Label files: {total_labels}",
        f"Valid boxes: {total_boxes}",
        f"Empty labels: {total_empty}",
        f"Annotation issues: {total_issues}",
        f"Exact duplicate image groups: {len(duplicate_groups)}",
        "",
        "DETAILS:",
        *report
    ]
    output.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Report saved: {output}")

if __name__ == "__main__":
    main()
