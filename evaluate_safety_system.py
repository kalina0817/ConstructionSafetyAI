import argparse
import csv
import json
from pathlib import Path

from evaluation_metrics import binary_metrics, cohen_kappa


TRUTH_VALUES = {
    "1": 1,
    "true": 1,
    "yes": 1,
    "present": 1,
    "0": 0,
    "false": 0,
    "no": 0,
    "absent": 0,
}
UNCERTAIN_VALUES = {
    "uncertain",
    "unassessable",
    "unknown",
    "not_assessable",
    "not assessable",
}
SUPPORTED_CATEGORIES = {
    "potential_missing_ppe",
    "potential_worker_machinery_interaction",
}


def load_gold_annotations(csv_path):
    annotations = {}
    raters = {
        "rater_1": [],
        "rater_2": [],
        "excluded_uncertain_pairs": 0,
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image_id", "finding_category", "ground_truth"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Gold CSV must contain image_id, finding_category, and "
                "ground_truth columns."
            )
        rater_columns = {"rater_1", "rater_2"} & set(reader.fieldnames)
        if rater_columns and len(rater_columns) != 2:
            raise ValueError(
                "Gold CSV must provide both rater_1 and rater_2 columns "
                "when inter-rater agreement is included."
            )
        has_raters = {"rater_1", "rater_2"}.issubset(reader.fieldnames)

        for line_number, row in enumerate(reader, start=2):
            image_id = Path(row["image_id"].strip()).name
            category = row["finding_category"].strip()
            label_text = row["ground_truth"].strip().lower()
            if not image_id or not category:
                raise ValueError(
                    f"Gold CSV row {line_number} has an empty image or category."
                )
            if category not in SUPPORTED_CATEGORIES:
                raise ValueError(
                    f"Unsupported finding_category on row {line_number}: "
                    f"{category!r}"
                )
            if (
                label_text not in TRUTH_VALUES
                and label_text not in UNCERTAIN_VALUES
            ):
                raise ValueError(
                    f"Invalid ground_truth on row {line_number}: "
                    f"{row['ground_truth']!r}"
                )

            key = (image_id, category)
            if key in annotations:
                raise ValueError(
                    f"Duplicate gold label for {image_id!r}, {category!r}."
                )
            annotations[key] = TRUTH_VALUES.get(label_text)

            if has_raters:
                paired_values = {}
                for rater in raters:
                    if rater == "excluded_uncertain_pairs":
                        continue
                    value = row[rater].strip().lower()
                    if (
                        value not in TRUTH_VALUES
                        and value not in UNCERTAIN_VALUES
                    ):
                        raise ValueError(
                            f"Invalid {rater} label on row {line_number}: "
                            f"{row[rater]!r}"
                        )
                    paired_values[rater] = TRUTH_VALUES.get(value)
                if all(value is not None for value in paired_values.values()):
                    for rater, value in paired_values.items():
                        raters[rater].append(value)
                else:
                    raters["excluded_uncertain_pairs"] += 1

    if not annotations:
        raise ValueError("Gold CSV has no annotation rows.")
    return annotations, raters if has_raters else {}


def load_system_predictions(reports_dir, required_images):
    predictions = {}
    for report_path in reports_dir.rglob("detections.json"):
        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
        assessment = report.get("safety_assessment")
        input_image = report.get("input_image")
        if (
            not isinstance(assessment, dict)
            or not isinstance(input_image, str)
            or not input_image
        ):
            continue

        image_id = Path(input_image).name
        if image_id not in required_images:
            continue
        if image_id in predictions:
            raise ValueError(
                f"Multiple evaluated run reports found for {image_id!r}; "
                "select a directory with exactly one run per image."
            )

        findings = assessment.get("findings")
        if not isinstance(findings, list):
            raise ValueError(
                f"Safety assessment findings are invalid in {report_path}."
            )
        if any(
            not isinstance(item, dict)
            or item.get("finding_category") not in SUPPORTED_CATEGORIES
            for item in findings
        ):
            raise ValueError(
                f"Safety assessment contains an invalid finding category "
                f"in {report_path}."
            )
        predictions[image_id] = {
            item["finding_category"] for item in findings
        }

    missing = sorted(required_images - predictions.keys())
    if missing:
        raise ValueError(
            "No compatible safety assessment report found for labeled "
            f"images: {missing[:20]}"
        )
    return predictions


def evaluate(gold_annotations, predictions):
    missing_predictions = {
        image_id for image_id, _ in gold_annotations
    } - predictions.keys()
    if missing_predictions:
        raise ValueError(
            "Predictions are missing labeled images: "
            f"{sorted(missing_predictions)[:20]}"
        )

    categories = sorted({category for _, category in gold_annotations})
    results = {}
    annotation_summary = {
        "total_pairs": len(gold_annotations),
        "evaluated_pairs": sum(
            label is not None for label in gold_annotations.values()
        ),
        "excluded_uncertain_pairs": sum(
            label is None for label in gold_annotations.values()
        ),
        "by_category": {},
    }
    evaluated_annotations = [
        (image_id, category, label)
        for (image_id, category), label in gold_annotations.items()
        if label is not None
    ]

    for category in categories:
        rows = [
            (image_id, label)
            for (image_id, row_category), label in gold_annotations.items()
            if row_category == category
        ]
        evaluated_rows = [
            (image_id, label) for image_id, label in rows
            if label is not None
        ]
        truth = [label for _, label in evaluated_rows]
        predicted = [
            int(category in predictions[image_id])
            for image_id, _ in evaluated_rows
        ]
        results[category] = binary_metrics(truth, predicted)
        annotation_summary["by_category"][category] = {
            "total_pairs": len(rows),
            "evaluated_pairs": len(evaluated_rows),
            "excluded_uncertain_pairs": len(rows) - len(evaluated_rows),
        }

    return {
        "schema_version": "1.0",
        "evaluation_unit": "image and hazard category",
        "annotation_summary": annotation_summary,
        "categories": results,
        "overall": binary_metrics(
            [label for _, _, label in evaluated_annotations],
            [
                int(category in predictions[image_id])
                for image_id, category, _ in evaluated_annotations
            ],
        ),
        "limitations": [
            "Metrics reflect only the categories and images in the supplied "
            "independent gold annotations.",
            "They do not measure localization quality or prove real-world "
            "hazard occurrence.",
            "Do not use the detector training, validation, or locked test "
            "images as an integrated-system tuning set.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate image-level contextual findings against independent "
            "gold labels."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.annotations.is_file():
        parser.error(f"annotation CSV not found: {args.annotations}")
    if not args.reports_dir.is_dir():
        parser.error(f"reports directory not found: {args.reports_dir}")

    annotations, raters = load_gold_annotations(args.annotations)
    predictions = load_system_predictions(
        args.reports_dir,
        {image_id for image_id, _ in annotations},
    )
    report = evaluate(annotations, predictions)

    if "rater_1" in raters and "rater_2" in raters:
        report["inter_rater_agreement"] = {
            "measure": "Cohen's kappa",
            "paired_labels": len(raters["rater_1"]),
            "excluded_uncertain_pairs": raters[
                "excluded_uncertain_pairs"
            ],
            "value": (
                cohen_kappa(raters["rater_1"], raters["rater_2"])
                if raters["rater_1"] else None
            ),
            "scope": "Binary image-level hazard-category annotations.",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Evaluation report written to: {args.output}")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
