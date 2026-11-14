import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline_paths import resolve_run_folder


def build_adjudicated_report(detections_report, review_report, run_folder):
    detections_file = run_folder / "detections.json"
    review_file = run_folder / "human_review.json"

    review_source = review_report.get("source_report")
    if not review_source:
        raise ValueError("Human review is missing its source_report.")

    if Path(review_source).resolve() != detections_file.resolve():
        raise ValueError(
            "The review file references a different source report."
        )

    candidates = detections_report.get(
        "review_analysis", {}
    ).get("candidates", [])
    decisions = review_report.get("review_decisions")

    if not isinstance(candidates, list) or not all(
        isinstance(candidate, dict) for candidate in candidates
    ):
        raise ValueError("Detection report candidates must be a list of objects.")
    if not isinstance(decisions, list):
        raise ValueError("Human review review_decisions must be a list.")

    allowed_decisions = {"Pending", "Confirmed", "Rejected", "Uncertain"}
    seen_indices = set()
    confirmed = []
    rejected = []
    uncertain = []
    pending = []

    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("Each human review decision must be an object.")
        index = item.get("candidate_index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(
                "Each review decision needs an integer candidate_index."
            )
        if index < 0 or index >= len(candidates):
            raise ValueError(
                f"Review references unknown candidate index: {index}"
            )
        if index in seen_indices:
            raise ValueError(
                f"Duplicate review decision for candidate {index}."
            )
        seen_indices.add(index)

        candidate = item.get("candidate")
        if not isinstance(candidate, dict) or candidate != candidates[index]:
            raise ValueError(
                f"Review candidate {index} does not match the source report."
            )

        decision = item.get("decision")
        if not isinstance(decision, str) or decision not in allowed_decisions:
            raise ValueError(
                f"Invalid decision for candidate {index}: {decision!r}"
            )

        finding = {
            "candidate_index": index,
            "candidate_type": candidate.get("candidate_type"),
            "class_name": candidate.get("class_name"),
            "detector": candidate.get("detector"),
            "confidence": candidate.get("confidence"),
            "bbox_xyxy": candidate.get("bbox_xyxy"),
            "person_class": candidate.get("person_class"),
            "machinery_class": candidate.get("machinery_class"),
            "person_bbox_xyxy": candidate.get("person_bbox_xyxy"),
            "machinery_bbox_xyxy": candidate.get("machinery_bbox_xyxy"),
            "normalized_image_distance": candidate.get(
                "normalized_image_distance"
            ),
            "human_decision": decision,
            "reviewer_notes": item.get("reviewer_notes", ""),
            "original_candidate": candidate,
        }

        if decision == "Confirmed":
            confirmed.append(finding)
        elif decision == "Rejected":
            rejected.append(finding)
        elif decision == "Uncertain":
            uncertain.append(finding)
        else:
            pending.append(finding)

    expected_indices = set(range(len(candidates)))
    if seen_indices != expected_indices:
        missing_indices = sorted(expected_indices - seen_indices)
        raise ValueError(
            "Human review is incomplete; missing candidate decisions: "
            f"{missing_indices}"
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_detection_report": str(detections_file),
        "source_human_review": str(review_file),
        "total_reviewed_candidates": len(decisions),
        "summary": {
            "confirmed_predictions": len(confirmed),
            "rejected_predictions": len(rejected),
            "uncertain_predictions": len(uncertain),
            "pending_predictions": len(pending),
        },
        "confirmed_findings": confirmed,
        "rejected_predictions": rejected,
        "uncertain_findings": uncertain,
        "pending_findings": pending,
        "interpretation_note": (
            "Confirmed findings reflect human review of the visible image. "
            "They are not automatically proof of a real-world safety incident. "
            "Rejected predictions are excluded from confirmed findings. "
            "Original detector outputs and review decisions are preserved "
            "in their separate source files."
        ),
    }


def main():
    run_folder = resolve_run_folder(
        "Adjudicate saved human decisions for an inference run."
    )
    detections_file = run_folder / "detections.json"
    review_file = run_folder / "human_review.json"
    output_file = run_folder / "adjudicated_findings.json"

    for path in (detections_file, review_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required report not found: {path}")

    with detections_file.open("r", encoding="utf-8") as file:
        detections_report = json.load(file)
    with review_file.open("r", encoding="utf-8") as file:
        review_report = json.load(file)

    report = build_adjudicated_report(
        detections_report, review_report, run_folder
    )
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Adjudicated report created.")
    print(f"Output: {output_file}")
    print("\nSummary:")
    for name, count in report["summary"].items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
