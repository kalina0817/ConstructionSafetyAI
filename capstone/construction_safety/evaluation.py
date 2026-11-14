import csv
import json
from pathlib import Path

from construction_safety.config import SCREENING_CATEGORIES


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


def binary_metrics(truth, predicted):
    if len(truth) != len(predicted):
        raise ValueError("Truth and prediction lengths must match.")
    if any(value not in (0, 1) for value in [*truth, *predicted]):
        raise ValueError("Binary labels must contain only 0 or 1.")
    tp = sum(actual == 1 and predicted_value == 1
             for actual, predicted_value in zip(truth, predicted))
    fp = sum(actual == 0 and predicted_value == 1
             for actual, predicted_value in zip(truth, predicted))
    fn = sum(actual == 1 and predicted_value == 0
             for actual, predicted_value in zip(truth, predicted))
    tn = sum(actual == 0 and predicted_value == 0
             for actual, predicted_value in zip(truth, predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "support": len(truth),
        "positive_support": sum(truth),
        "negative_support": len(truth) - sum(truth),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": (tp + tn) / len(truth) if truth else None,
        "f1": f1,
    }


def cohen_kappa(first_rater, second_rater):
    if len(first_rater) != len(second_rater):
        raise ValueError("Rater label lengths must match.")
    if not first_rater:
        return None
    if any(value not in (0, 1) for value in [*first_rater, *second_rater]):
        raise ValueError("Rater labels must contain only 0 or 1.")
    count = len(first_rater)
    observed = sum(
        first == second
        for first, second in zip(first_rater, second_rater)
    ) / count
    first_positive = sum(first_rater) / count
    second_positive = sum(second_rater) / count
    expected = (
        first_positive * second_positive
        + (1 - first_positive) * (1 - second_positive)
    )
    return None if expected == 1 else (observed - expected) / (1 - expected)


def load_annotations(csv_path):
    annotations = {}
    raters = None
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image_id", "finding_category", "ground_truth"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV requires image_id, finding_category, and ground_truth."
            )
        rater_columns = {"rater_1", "rater_2"} & set(reader.fieldnames)
        if rater_columns and len(rater_columns) != 2:
            raise ValueError("Include both rater_1 and rater_2 or neither.")
        has_raters = len(rater_columns) == 2
        rater_rows = {"rater_1": [], "rater_2": []}
        excluded_rater_pairs = 0

        for line_number, row in enumerate(reader, start=2):
            image_id = Path(row["image_id"].strip()).name
            category = row["finding_category"].strip()
            label = row["ground_truth"].strip().lower()
            if not image_id:
                raise ValueError(f"Empty image_id on CSV row {line_number}.")
            if category not in SCREENING_CATEGORIES:
                raise ValueError(
                    f"Unsupported category on CSV row {line_number}: "
                    f"{category!r}"
                )
            if label not in TRUTH_VALUES and label not in UNCERTAIN_VALUES:
                raise ValueError(
                    f"Invalid ground_truth on CSV row {line_number}: "
                    f"{label!r}"
                )
            key = image_id, category
            if key in annotations:
                raise ValueError(f"Duplicate annotation row: {key!r}")
            annotations[key] = TRUTH_VALUES.get(label)
            if has_raters:
                values = {}
                for rater in rater_rows:
                    value = row[rater].strip().lower()
                    if value not in TRUTH_VALUES and value not in UNCERTAIN_VALUES:
                        raise ValueError(
                            f"Invalid {rater} value on row {line_number}."
                        )
                    values[rater] = TRUTH_VALUES.get(value)
                if all(value is not None for value in values.values()):
                    for rater, value in values.items():
                        rater_rows[rater].append(value)
                else:
                    excluded_rater_pairs += 1

    if not annotations:
        raise ValueError("The annotation CSV has no rows.")
    if has_raters:
        raters = {
            **rater_rows,
            "excluded_uncertain_pairs": excluded_rater_pairs,
        }
    return annotations, raters


def load_predictions(reports_dir, required_images):
    reports_dir = Path(reports_dir)
    predictions = {}
    for report_path in reports_dir.rglob("detections.json"):
        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
        image_id = report.get("input_image")
        assessment = report.get("safety_assessment")
        if (
            not isinstance(image_id, str)
            or not isinstance(assessment, dict)
            or Path(image_id).name not in required_images
        ):
            continue
        image_id = Path(image_id).name
        if image_id in predictions:
            raise ValueError(
                f"Multiple run reports found for {image_id!r}; use a "
                "directory with exactly one run per image."
            )
        findings = assessment.get("findings")
        if not isinstance(findings, list):
            raise ValueError(f"Invalid assessment findings in {report_path}.")
        categories = set()
        for finding in findings:
            category = (
                finding.get("finding_category")
                if isinstance(finding, dict) else None
            )
            if category not in SCREENING_CATEGORIES:
                raise ValueError(f"Invalid finding category in {report_path}.")
            categories.add(category)
        predictions[image_id] = categories
    missing = required_images - predictions.keys()
    if missing:
        raise ValueError(
            f"No compatible report for labeled images: {sorted(missing)[:20]}"
        )
    return predictions


def evaluate(annotations, predictions):
    missing_images = {image for image, _ in annotations} - predictions.keys()
    if missing_images:
        raise ValueError(f"Missing predictions for {sorted(missing_images)}")

    results = {}
    category_counts = {}
    evaluated = [
        (image, category, label)
        for (image, category), label in annotations.items()
        if label is not None
    ]
    for category in sorted({category for _, category in annotations}):
        rows = [
            (image, label)
            for (image, row_category), label in annotations.items()
            if row_category == category
        ]
        binary_rows = [(image, label) for image, label in rows if label is not None]
        results[category] = binary_metrics(
            [label for _, label in binary_rows],
            [int(category in predictions[image]) for image, _ in binary_rows],
        )
        category_counts[category] = {
            "total_pairs": len(rows),
            "evaluated_pairs": len(binary_rows),
            "excluded_uncertain_pairs": len(rows) - len(binary_rows),
        }
    return {
        "schema_version": "1.0",
        "evaluation_unit": "image and screening category",
        "annotation_summary": {
            "total_pairs": len(annotations),
            "evaluated_pairs": len(evaluated),
            "excluded_uncertain_pairs": len(annotations) - len(evaluated),
            "by_category": category_counts,
        },
        "categories": results,
        "overall": binary_metrics(
            [label for _, _, label in evaluated],
            [
                int(category in predictions[image])
                for image, category, _ in evaluated
            ],
        ),
        "limitations": [
            "Metrics apply only to the independently labeled images and "
            "categories provided.",
            "The evaluator does not measure localization or prove a "
            "real-world hazard.",
            "Do not use the test set for model selection or threshold tuning.",
        ],
    }
