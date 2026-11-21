import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path.cwd()
ANNOTATIONS = (
    PROJECT / "research" / "independent_scenario_evaluation"
    / "independent_human_annotations_clean.csv"
)
OUTPUT_ROOT = PROJECT / "reports" / "reasoning_model_comparison"

MODEL_NAMES = {
    "minicpm-v_latest": "MiniCPM",
    "qwen3-vl_4b-instruct": "Qwen",
}

CATEGORIES = [
    "potential_missing_ppe",
    "potential_worker_machinery_interaction",
]


def read_annotations():
    result = {}
    with ANNOTATIONS.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["image_id"].strip(), row["finding_category"].strip())
            label = row["ground_truth"].strip().lower()

            if label not in {"0", "1", "uncertain"}:
                raise ValueError(f"Invalid human label: {key}={label}")

            if key in result:
                raise ValueError(f"Duplicate annotation: {key}")

            result[key] = label

    return result


def extract_text(result):
    for key in (
        "detector_evidence_assessment",
        "reasoning",
        "explanation",
        "assessment",
        "rationale",
        "summary",
    ):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def pct(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_category(rows):
    evaluable = [
        row for row in rows
        if row["ground_truth"] in {"0", "1"}
    ]

    tp = fp = tn = fn = 0
    abstain_positive = abstain_negative = 0
    invalid = []

    for row in evaluable:
        gt = row["ground_truth"]
        pred = row["decision"]

        if pred == "uncertain":
            if gt == "1":
                abstain_positive += 1
            else:
                abstain_negative += 1
        elif pred == "supported":
            if gt == "1":
                tp += 1
            else:
                fp += 1
        elif pred == "not_supported":
            if gt == "0":
                tn += 1
            else:
                fn += 1
        else:
            invalid.append({
                "image": row["image_id"],
                "decision": pred,
            })

    total = len(evaluable)
    positive_total = tp + fn + abstain_positive
    negative_total = tn + fp + abstain_negative
    covered = tp + fp + tn + fn

    return {
        "evaluated_labels": total,
        "positive_labels": positive_total,
        "negative_labels": negative_total,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "positive_abstentions": abstain_positive,
        "negative_abstentions": abstain_negative,
        "decisive_predictions": covered,
        "coverage": pct(covered, total),
        "coverage_adjusted_accuracy": pct(tp + tn, total),
        "selective_accuracy": pct(tp + tn, covered),
        "coverage_adjusted_sensitivity": pct(tp, positive_total),
        "coverage_adjusted_specificity": pct(tn, negative_total),
        "selective_sensitivity": pct(tp, tp + fn),
        "selective_specificity": pct(tn, tn + fp),
        "invalid_decisions": invalid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="20261120T205556Z",
        help="Reasoning model comparison run ID",
    )
    args = parser.parse_args()

    raw_root = OUTPUT_ROOT / "raw_outputs" / args.run_id
    if not raw_root.is_dir():
        raise SystemExit(f"STOP: Run folder not found: {raw_root}")

    if not ANNOTATIONS.is_file():
        raise SystemExit(f"STOP: Annotation CSV not found: {ANNOTATIONS}")

    annotations = read_annotations()
    records = []

    for path in sorted(raw_root.glob("*/*.json")):
        model_folder = path.parent.name
        if model_folder not in MODEL_NAMES:
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        image_id = data.get("image_id")

        if not image_id:
            raise ValueError(f"Missing image_id in {path}")

        parsed = data.get("parsed_response") or {}

        for category in CATEGORIES:
            result = parsed.get(category) or {}
            if not isinstance(result, dict):
                result = {}

            decision = str(result.get("decision", "missing")).strip().lower()
            evidence = result.get("visual_evidence", [])
            if not isinstance(evidence, list):
                evidence = []

            records.append({
                "image_id": image_id,
                "category": category,
                "model": MODEL_NAMES[model_folder],
                "ground_truth": annotations.get(
                    (image_id, category), "not_annotated"
                ),
                "decision": decision,
                "visual_evidence_count": len(evidence),
                "structured_evidence_present": bool(evidence),
                "reasoning_text": extract_text(result),
                "visual_evidence_json": json.dumps(
                    evidence, ensure_ascii=False
                ),
                "category_response_json": json.dumps(
                    result, ensure_ascii=False
                ),
                "source_file": str(path.relative_to(PROJECT)),
            })

    if not records:
        raise SystemExit("STOP: No model responses were loaded.")

    keys = [
        (r["image_id"], r["category"], r["model"])
        for r in records
    ]
    if len(keys) != len(set(keys)):
        raise SystemExit("STOP: Duplicate image/category/model results found.")

    models_found = sorted({r["model"] for r in records})
    metrics = {}
    for model in models_found:
        metrics[model] = {}
        for category in CATEGORIES:
            subset = [
                r for r in records
                if r["model"] == model and r["category"] == category
            ]
            metrics[model][category] = evaluate_category(subset)

    evidence_summary = {}
    for model in models_found:
        subset = [r for r in records if r["model"] == model]
        evidence_summary[model] = {
            "responses": len(subset),
            "responses_with_structured_evidence": sum(
                r["structured_evidence_present"] for r in subset
            ),
            "responses_with_empty_structured_evidence": sum(
                not r["structured_evidence_present"] for r in subset
            ),
        }

    annotation_counts = {}
    for category in CATEGORIES:
        counts = Counter(
            label for (image, cat), label in annotations.items()
            if cat == category
        )
        annotation_counts[category] = {
            "positive": counts["1"],
            "negative": counts["0"],
            "uncertain": counts["uncertain"],
        }

    output_dir = OUTPUT_ROOT / f"evaluation_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "reasoning_model_metrics.csv"
    cases_csv = output_dir / "reasoning_model_case_comparison.csv"
    report_json = output_dir / "reasoning_model_evaluation_report.json"

    metric_fields = [
        "model", "category", "evaluated_labels",
        "positive_labels", "negative_labels",
        "true_positive", "false_positive",
        "true_negative", "false_negative",
        "positive_abstentions", "negative_abstentions",
        "decisive_predictions", "coverage",
        "coverage_adjusted_accuracy", "selective_accuracy",
        "coverage_adjusted_sensitivity",
        "coverage_adjusted_specificity",
        "selective_sensitivity", "selective_specificity",
    ]

    with metrics_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        for model in models_found:
            for category in CATEGORIES:
                row = {
                    "model": model,
                    "category": category,
                    **metrics[model][category],
                }
                writer.writerow({
                    key: row.get(key)
                    for key in metric_fields
                })

    case_fields = [
        "image_id", "category", "model", "ground_truth",
        "decision", "visual_evidence_count",
        "structured_evidence_present", "reasoning_text",
        "visual_evidence_json", "category_response_json",
        "source_file",
    ]

    with cases_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=case_fields)
        writer.writeheader()
        writer.writerows(records)

    report = {
        "run_id": args.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_file": str(ANNOTATIONS.relative_to(PROJECT)),
        "raw_output_directory": str(raw_root.relative_to(PROJECT)),
        "evaluation_rules": {
            "supported": "positive prediction",
            "not_supported": "negative prediction",
            "uncertain": "model abstention",
            "human_uncertain": "excluded from binary metrics",
            "coverage_adjusted_metrics":
                "abstentions count as failures to classify correctly",
            "selective_metrics":
                "calculated only among decisive predictions",
        },
        "annotation_counts": annotation_counts,
        "metrics": metrics,
        "structured_evidence_summary": evidence_summary,
        "case_count": len(records),
        "models": models_found,
    }

    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nEVALUATION COMPLETE")
    print(f"Run: {args.run_id}")
    print(f"Cases: {len(records)}")
    print(f"Output folder: {output_dir}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Case comparison CSV: {cases_csv}")
    print(f"JSON report: {report_json}")

    print("\nSTRUCTURED EVIDENCE")
    for model, values in evidence_summary.items():
        print(
            f"{model}: "
            f"{values['responses_with_structured_evidence']}/"
            f"{values['responses']} responses contain evidence"
        )


if __name__ == "__main__":
    main()
