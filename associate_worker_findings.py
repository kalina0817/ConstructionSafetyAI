
import json
from pathlib import Path
from datetime import datetime, timezone

from pipeline_paths import resolve_run_folder
from ppe_safety_rules import RECOMMENDATION_RULES


# Minimum fraction of a PPE prediction's area that must fall
# inside a detected person's bounding box.
CONTAINMENT_THRESHOLD = 0.80


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def containment_ratio(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    x1 = max(ix1, ox1)
    y1 = max(iy1, oy1)
    x2 = min(ix2, ox2)
    y2 = min(iy2, oy2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = box_area(inner_box)

    if area == 0:
        return 0.0

    return intersection / area


def find_source_detection(finding, detections):
    """Match a reviewed finding to its original detector output."""

    matches = [
        detection for detection in detections
        if (
            detection.get("detector") == finding.get("detector")
            and detection.get("class_name") == finding.get("class_name")
            and abs(
                detection.get("confidence", 0)
                - finding.get("confidence", 0)
            ) < 0.00001
        )
    ]

    reviewed_box = finding.get("box_xyxy")
    if reviewed_box is not None:
        matches = [
            detection for detection in matches
            if detection.get("box_xyxy") == reviewed_box
        ]

    return matches[0] if len(matches) == 1 else None


def main():
    run_folder = resolve_run_folder(
        "Associate confirmed PPE findings with detected workers."
    )
    detections_file = run_folder / "detections.json"
    adjudicated_file = run_folder / "adjudicated_findings.json"
    output_file = run_folder / "worker_level_findings.json"

    for path in (detections_file, adjudicated_file):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with detections_file.open("r", encoding="utf-8") as file:
        detection_report = json.load(file)

    with adjudicated_file.open("r", encoding="utf-8") as file:
        adjudicated = json.load(file)

    detections = detection_report.get("detections", [])

    people = [
        (index, detection)
        for index, detection in enumerate(detections)
        if detection.get("class_name") == "Person"
        and detection.get("box_xyxy")
    ]

    grouped = {}
    unassigned = []

    confirmed_findings = adjudicated.get("confirmed_findings", [])

    for finding in confirmed_findings:
        class_name = finding.get("class_name")

        # This association step handles PPE findings only.
        if class_name not in RECOMMENDATION_RULES:
            unassigned.append({
                "finding": finding,
                "reason": "No worker-association rule for this finding type."
            })
            continue

        source = find_source_detection(finding, detections)

        if source is None or not source.get("box_xyxy"):
            unassigned.append({
                "finding": finding,
                "reason": (
                    "Original detection or an unambiguous bounding-box "
                    "match was not found."
                )
            })
            continue

        ppe_box = source["box_xyxy"]
        best_person = None
        best_ratio = 0.0

        for person_index, person in people:
            ratio = containment_ratio(
                ppe_box,
                person["box_xyxy"]
            )

            if ratio > best_ratio:
                best_ratio = ratio
                best_person = (person_index, person)

        if (
            best_person is None
            or best_ratio < CONTAINMENT_THRESHOLD
        ):
            unassigned.append({
                "finding": finding,
                "source_detection": source,
                "best_containment_ratio": best_ratio,
                "reason": "No sufficiently supported person association."
            })
            continue

        person_index, person = best_person
        group_key = (person_index, class_name)

        if group_key not in grouped:
            grouped[group_key] = {
                "worker_detection_index": person_index,
                "worker_confidence": person["confidence"],
                "worker_box_xyxy": person["box_xyxy"],
                "ppe_category": class_name,
                "human_decision": "Confirmed",
                "original_predictions": []
            }

        grouped[group_key]["original_predictions"].append({
            "candidate_index": finding.get("candidate_index"),
            "confidence": finding.get("confidence"),
            "detector": finding.get("detector"),
            "box_xyxy": source["box_xyxy"],
            "containment_ratio": round(best_ratio, 5),
            "reviewer_notes": finding.get("reviewer_notes", "")
        })

    worker_findings = []

    for group in grouped.values():
        predictions = group["original_predictions"]

        worker_findings.append({
            **group,
            "prediction_count": len(predictions),
            "confidence_values": [
                item["confidence"] for item in predictions
            ],
            "recommendation_eligible": True
        })

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_detection_report": str(detections_file),
        "source_adjudicated_report": str(adjudicated_file),
        "association_method": (
            "PPE bounding-box containment within detected person boxes."
        ),
        "containment_threshold": CONTAINMENT_THRESHOLD,
        "person_count": len(people),
        "confirmed_prediction_count": len(confirmed_findings),
        "worker_level_finding_count": len(worker_findings),
        "worker_level_findings": worker_findings,
        "unassigned_findings": unassigned,
        "interpretation_note": (
            "Grouped findings are associations based on image bounding "
            "boxes, not verified worker identities. Original predictions "
            "and confidence scores are retained. A group represents one "
            "worker-level review category, not necessarily a separate "
            "safety incident."
        )
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Worker-level findings report created.")
    print(f"Output: {output_file}")
    print(f"Detected people: {len(people)}")
    print(f"Confirmed PPE predictions: {len(confirmed_findings)}")
    print(f"Worker-level findings: {len(worker_findings)}")
    print(f"Unassigned findings: {len(unassigned)}")

    print("\nGrouped findings:")
    for item in worker_findings:
        print(
            f"{item['ppe_category']}: "
            f"{item['prediction_count']} prediction(s), "
            f"worker detection #{item['worker_detection_index']}"
        )


if __name__ == "__main__":
    main()
