
import json
from datetime import datetime, timezone

from pipeline_paths import resolve_run_folder
from ppe_safety_rules import RECOMMENDATION_RULES


def build_worker_recommendations(worker_findings):
    recommendations = []

    for finding in worker_findings:
        class_name = finding.get("ppe_category")
        rule = RECOMMENDATION_RULES.get(class_name)

        recommendation = {
            "worker_detection_index": finding.get(
                "worker_detection_index"
            ),
            "worker_confidence": finding.get("worker_confidence"),
            "worker_bbox_xyxy": finding.get("worker_bbox_xyxy"),
            "ppe_category": class_name,
            "human_decision": finding.get("human_decision"),
            "prediction_count": finding.get("prediction_count"),
            "supporting_predictions": finding.get(
                "original_predictions", []
            )
        }

        if rule is None:
            recommendation.update({
                "recommendation_status": "requires_rule_review",
                "observation": (
                    "A worker-level finding was confirmed, but no "
                    "recommendation rule is defined for this category."
                ),
                "recommendation": None,
                "limitation": None
            })
        else:
            recommendation.update({
                "recommendation_status": "recommendation_generated",
                "observation": rule["observation"],
                "recommendation": rule["recommendation"],
                "limitation": rule["limitation"]
            })

        recommendations.append(recommendation)

    return recommendations


def main():
    run_folder = resolve_run_folder(
        "Generate cautious recommendations from reviewed findings."
    )
    detections_file = run_folder / "detections.json"
    worker_findings_file = run_folder / "worker_level_findings.json"
    adjudicated_file = run_folder / "adjudicated_findings.json"
    output_file = run_folder / "safety_reasoning.json"

    for path in (detections_file, worker_findings_file, adjudicated_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Required report not found: {path}"
            )

    with detections_file.open("r", encoding="utf-8") as file:
        detection_report = json.load(file)
    with worker_findings_file.open(
        "r", encoding="utf-8"
    ) as file:
        worker_report = json.load(file)

    with adjudicated_file.open(
        "r", encoding="utf-8"
    ) as file:
        adjudicated = json.load(file)

    worker_findings = worker_report.get(
        "worker_level_findings", []
    )

    rejected = adjudicated.get("rejected_predictions", [])
    uncertain = adjudicated.get("uncertain_findings", [])
    pending = adjudicated.get("pending_findings", [])

    recommendations = build_worker_recommendations(
        worker_findings
    )

    generated_count = sum(
        1
        for item in recommendations
        if item["recommendation_status"] == "recommendation_generated"
    )

    reasoning_report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_worker_findings_report": str(worker_findings_file),
        "source_adjudicated_report": str(adjudicated_file),
        "review_summary": adjudicated.get("summary", {}),
        "worker_level_summary": {
            "detected_people": worker_report.get("person_count", 0),
            "confirmed_predictions": worker_report.get(
                "confirmed_prediction_count", 0
            ),
            "worker_level_findings": len(worker_findings),
            "unassigned_findings": len(
                worker_report.get("unassigned_findings", [])
            )
        },
        "safety_recommendations": recommendations,
        "contextual_screening": detection_report.get("safety_assessment"),
        "rejected_predictions": rejected,
        "uncertain_findings": uncertain,
        "pending_findings": pending,
        "interpretation_note": (
            "Recommendations are generated from human-confirmed "
            "worker-level findings. Multiple predictions associated "
            "with the same worker and PPE category are grouped into "
            "one finding, while all original predictions are preserved "
            "as supporting evidence. Contextual screening findings remain "
            "unconfirmed until reviewed. These recommendations are screening "
            "guidance, not formal safety certification, regulatory "
            "determinations, or proof of an unsafe condition. Site policies, "
            "work activities, and applicable requirements must be verified."
        )
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(reasoning_report, file, indent=4)

    print("Worker-level safety reasoning report created.")
    print(f"Output: {output_file}")

    print("\nWorker-level summary:")
    for category, count in reasoning_report[
        "worker_level_summary"
    ].items():
        print(f"{category}: {count}")

    print("\nReasoning results:")
    print(f"Recommendations generated: {generated_count}")
    print(f"Rejected predictions retained: {len(rejected)}")
    print(f"Uncertain findings retained: {len(uncertain)}")
    print(f"Pending findings retained: {len(pending)}")


if __name__ == "__main__":
    main()
