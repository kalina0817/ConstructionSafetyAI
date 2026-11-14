"""
Prepare separate, leakage-aware PPE and construction-vehicle datasets.

- Reads only data/raw/
- Writes new copies under data/processed/
- Leaves all source images, labels, and YAML files unchanged
- Keeps PPE's supplied train/valid/test splits
- For vehicles, keeps valid and deterministically reserves 10% of unique train images for test
- Removes exact duplicate image copies from processed output, keeping one occurrence
- Writes reports under data/reports/
- Does NOT train any model
"""
from __future__ import annotations

import csv
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "data" / "reports"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VEHICLE_TEST_FRACTION = 0.10
SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not isinstance(names, list) or not names:
        raise ValueError(f"No class names found in {path}")
    return {"names": [str(n) for n in names]}


def get_split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path]:
    return dataset_root / split / "images", dataset_root / split / "labels"


def gather_images(dataset_root: Path, splits: list[str]) -> list[dict]:
    rows = []
    for split in splits:
        image_dir, label_dir = get_split_dirs(dataset_root, split)
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        for image in sorted(image_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
                continue
            label = label_dir / f"{image.stem}.txt"
            rows.append({
                "dataset": dataset_root.name,
                "split": split,
                "image": image,
                "label": label,
                "hash": sha256(image),
                "label_exists": label.is_file(),
            })
    return rows


def dedupe_and_assign(rows: list[dict], dataset_name: str, vehicle: bool) -> tuple[list[dict], list[dict]]:
    """
    Group exact-identical images, keeping one image per hash.
    PPE keeps original split priority test > valid > train.
    Vehicles keep valid as validation; unique train images are split train/test.
    """
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_hash[row["hash"]].append(row)

    chosen = []
    duplicate_audit = []
    for digest, group in by_hash.items():
        for row in group:
            duplicate_audit.append({
                "dataset": dataset_name,
                "hash": digest,
                "original_split": row["split"],
                "image": row["image"].name,
                "path": str(row["image"]),
                "group_size": len(group),
                "kept": "no",
                "processed_split": "",
            })

        if vehicle:
            # Keep validation-source images in validation if the same image is present there.
            priority = {"valid": 0, "test": 1, "train": 2}
        else:
            priority = {"test": 0, "valid": 1, "train": 2}
        group_sorted = sorted(group, key=lambda r: (priority.get(r["split"], 9), str(r["image"])))
        keep = group_sorted[0].copy()
        keep["source_rows"] = group
        chosen.append(keep)
        # Mark chosen record in audit.
        for audit in reversed(duplicate_audit):
            if audit["hash"] == digest and audit["path"] == str(keep["image"]):
                audit["kept"] = "yes"
                break

    if vehicle:
        valid_rows = [r for r in chosen if r["split"] == "valid"]
        # Existing test does not exist in this source. Only source-train items are candidates
        # for train/test; deterministic selection by SHA-256 of image hash + seed.
        candidates = [r for r in chosen if r["split"] != "valid"]
        candidates.sort(key=lambda r: hashlib.sha256(f"{SEED}:{r['hash']}".encode()).hexdigest())
        test_count = max(1, round(len(candidates) * VEHICLE_TEST_FRACTION)) if candidates else 0
        test_hashes = {r["hash"] for r in candidates[:test_count]}
        for row in chosen:
            if row["split"] == "valid":
                row["processed_split"] = "valid"
            elif row["hash"] in test_hashes:
                row["processed_split"] = "test"
            else:
                row["processed_split"] = "train"
    else:
        for row in chosen:
            row["processed_split"] = row["split"]

    # Update audit processed split for kept image.
    chosen_by_hash = {r["hash"]: r for r in chosen}
    for audit in duplicate_audit:
        kept = chosen_by_hash[audit["hash"]]
        audit["processed_split"] = kept["processed_split"]

    return chosen, duplicate_audit


def validate_yolo_label(label_path: Path, class_count: int) -> tuple[int, int]:
    """Return (valid line count, invalid line count); accepts bbox rows only."""
    if not label_path.exists():
        return 0, 1
    valid = invalid = 0
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return 0, 1
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(parts[0])
            nums = [float(v) for v in parts[1:]]
        except Exception:
            invalid += 1
            continue
        if cls < 0 or cls >= class_count or len(nums) != 4:
            invalid += 1
            continue
        x, y, w, h = nums
        if not all(0 <= v <= 1 for v in nums) or w <= 0 or h <= 0 or x - w / 2 < -1e-6 or y - h / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6:
            invalid += 1
            continue
        valid += 1
    return valid, invalid


def write_dataset(dataset_name: str, output_name: str, class_names: list[str], chosen: list[dict]) -> list[dict]:
    out_root = PROCESSED / output_name
    # Refuse to overwrite a previous processed dataset automatically.
    if out_root.exists():
        raise FileExistsError(
            f"Output already exists: {out_root}\n"
            "Move/rename it yourself if you want a fresh run; this script will not overwrite it."
        )
    out_root.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        (out_root / split / "images").mkdir(parents=True)
        (out_root / split / "labels").mkdir(parents=True)

    stats = []
    for row in chosen:
        split = row["processed_split"]
        target_image = out_root / split / "images" / row["image"].name
        target_label = out_root / split / "labels" / f"{row['image'].stem}.txt"
        if not row["label"].is_file():
            raise FileNotFoundError(f"Missing label for {row['image']}: {row['label']}")
        shutil.copy2(row["image"], target_image)
        shutil.copy2(row["label"], target_label)
        valid_lines, invalid_lines = validate_yolo_label(target_label, len(class_names))
        stats.append({
            "dataset": dataset_name,
            "processed_split": split,
            "image": row["image"].name,
            "label": target_label.name,
            "label_lines_valid": valid_lines,
            "label_lines_invalid": invalid_lines,
        })

    yaml_data = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with (out_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)
    return stats


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    configs = [
        ("construction-ppe", "ppe_detector", ["train", "valid", "test"], False),
        ("construction-vehicle-detection", "vehicle_detector", ["train", "valid"], True),
    ]

    all_dups = []
    all_stats = []
    summary = []

    for source_name, output_name, splits, is_vehicle in configs:
        source_root = RAW / source_name
        yaml_path = source_root / "data.yaml"
        if not source_root.is_dir() or not yaml_path.is_file():
            raise FileNotFoundError(f"Missing dataset or data.yaml: {source_root}")
        class_names = read_yaml(yaml_path)["names"]
        rows = gather_images(source_root, splits)
        if not rows:
            raise RuntimeError(f"No images found for {source_name}")
        missing_labels = [r for r in rows if not r["label_exists"]]
        if missing_labels:
            raise RuntimeError(
                f"{source_name}: {len(missing_labels)} images lack matching labels. "
                f"Example: {missing_labels[0]['image']}"
            )
        chosen, dups = dedupe_and_assign(rows, source_name, is_vehicle)
        stats = write_dataset(source_name, output_name, class_names, chosen)
        all_dups.extend(dups)
        all_stats.extend(stats)
        for split in ("train", "valid", "test"):
            split_rows = [r for r in chosen if r["processed_split"] == split]
            summary.append({
                "dataset": source_name,
                "output": output_name,
                "split": split,
                "images": len(split_rows),
                "class_count": len(class_names),
            })
        print(f"{source_name}: {len(rows)} source images -> {len(chosen)} unique images")
        print(f"Processed output: {PROCESSED / output_name}")

    with (REPORTS / "processed_dataset_split_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "output", "split", "images", "class_count"])
        writer.writeheader()
        writer.writerows(summary)

    with (REPORTS / "processed_dataset_duplicate_audit.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "hash", "original_split", "image", "path", "group_size", "kept", "processed_split"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_dups)

    with (REPORTS / "processed_dataset_label_validation.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "processed_split", "image", "label", "label_lines_valid", "label_lines_invalid"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_stats)

    invalid_total = sum(int(r["label_lines_invalid"]) for r in all_stats)
    print("\nPREPARATION COMPLETE — NO TRAINING WAS RUN")
    print(f"Summary: {REPORTS / 'processed_dataset_split_summary.csv'}")
    print(f"Duplicate audit: {REPORTS / 'processed_dataset_duplicate_audit.csv'}")
    print(f"Label validation: {REPORTS / 'processed_dataset_label_validation.csv'}")
    print(f"Invalid YOLO label lines found: {invalid_total}")
    if invalid_total:
        print("IMPORTANT: inspect the label-validation report before training.")


if __name__ == "__main__":
    main()
