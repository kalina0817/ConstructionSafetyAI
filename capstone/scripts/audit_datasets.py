import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from construction_safety.config import (
    MODEL_SPECS,
    PROJECT_ROOT,
    WORKSPACE_ROOT,
)
from construction_safety.dataset_audit import audit_dataset


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit configured detector datasets without modifying images, "
            "labels, or splits."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS) + ["all"],
        default="all",
    )
    args = parser.parse_args()
    selected = (
        sorted(MODEL_SPECS)
        if args.model == "all" else [args.model]
    )
    output_root = PROJECT_ROOT / "reports" / "dataset_audits"
    output_root.mkdir(parents=True, exist_ok=True)
    failed = False

    for model_name in selected:
        dataset_yaml = WORKSPACE_ROOT / MODEL_SPECS[model_name]["dataset"]
        report = audit_dataset(dataset_yaml)
        report_path = output_root / f"{model_name}_dataset_audit.json"
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        total_images = sum(
            split.get("image_count", 0)
            for split in report["splits"].values()
        )
        print(
            f"{model_name}: {total_images} images; "
            f"{report['error_count']} structural errors; "
            f"report: {report_path}"
        )
        for split_name, split_report in report["splits"].items():
            print(
                f"  {split_name}: {split_report['image_count']} images, "
                f"{split_report['annotation_count']} boxes, "
                f"{split_report['images_without_label_file']} images without "
                "a label file"
            )
        if report["errors"]:
            failed = True
            for error in report["errors"][:20]:
                print(f"  ERROR: {error}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
