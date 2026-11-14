import json
from pathlib import Path


DECISIONS = {"Pending", "Confirmed", "Rejected", "Uncertain"}
REVIEW_DISPOSITIONS = {
    "review_required",
    "no_review_required",
    "insufficient_evidence",
}
QUEUE_DISPOSITIONS = {
    "review_required",
    "insufficient_evidence",
}


def _finding_ids(report):
    findings = report.get("safety_assessment", {}).get("findings")

    if not isinstance(findings, list):
        raise ValueError("Report has no valid safety assessment findings.")

    finding_ids = [
        finding.get("finding_id")
        for finding in findings
        if isinstance(finding, dict)
    ]

    if len(finding_ids) != len(findings) or any(
        not isinstance(finding_id, str) or not finding_id
        for finding_id in finding_ids
    ):
        raise ValueError("Every finding must have a unique non-empty ID.")

    if len(set(finding_ids)) != len(finding_ids):
        raise ValueError("Report contains duplicate finding IDs.")

    return finding_ids


def get_automatic_review_queue(report):
    """Derive the review queue from automatic finding dispositions."""
    findings = report.get("safety_assessment", {}).get("findings")

    if not isinstance(findings, list):
        raise ValueError("Report has no valid safety assessment findings.")

    queue = []

    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("Every finding must be an object.")

        finding_id = finding.get("finding_id")
        disposition = finding.get("review_disposition")

        # Missing or unknown dispositions must never silently bypass review.
        if disposition not in REVIEW_DISPOSITIONS:
            queue.append(finding_id)
        elif disposition in QUEUE_DISPOSITIONS:
            queue.append(finding_id)

    return queue


def _pending_decisions(queue):
    return [
        {
            "finding_id": finding_id,
            "decision": "Pending",
            "reviewer_notes": "",
        }
        for finding_id in queue
    ]


def _validate_decision_rows(decisions, allowed_ids):
    if not isinstance(decisions, list):
        raise ValueError("Review decisions must be a list.")

    decision_ids = []

    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError("Every review decision must be an object.")

        finding_id = row.get("finding_id")

        if finding_id not in allowed_ids:
            raise ValueError(
                f"Review references unknown or unqueued finding: {finding_id!r}"
            )

        if row.get("decision") not in DECISIONS:
            raise ValueError(
                f"Invalid review decision for {finding_id!r}."
            )

        if not isinstance(row.get("reviewer_notes", ""), str):
            raise ValueError(
                f"Reviewer notes for {finding_id!r} must be text."
            )

        decision_ids.append(finding_id)

    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("A finding has more than one review decision.")

    return decision_ids


def build_initial_review(report, report_path):
    finding_ids = _finding_ids(report)
    queue = get_automatic_review_queue(report)

    return {
        "schema_version": "2.0",
        "source_report": str(Path(report_path).resolve()),
        "review_queue": queue,
        "review_decisions": _pending_decisions(queue),
        "not_queued_finding_ids": [
            finding_id
            for finding_id in finding_ids
            if finding_id not in set(queue)
        ],
    }


def validate_review(report, review):
    expected_ids = _finding_ids(report)
    version = review.get("schema_version")

    if version == "1.0":
        decisions = review.get("review_decisions")
        decision_ids = _validate_decision_rows(
            decisions,
            set(expected_ids),
        )

        if set(decision_ids) != set(expected_ids):
            raise ValueError(
                "Review is incomplete or does not match the current findings."
            )
        return

    if version != "2.0":
        raise ValueError("Unsupported human-review schema version.")

    automatic_queue = get_automatic_review_queue(report)
    queue = review.get("review_queue")
    not_queued = review.get("not_queued_finding_ids")

    if not isinstance(queue, list) or not isinstance(not_queued, list):
        raise ValueError("Review queue and not-queued IDs must be lists.")

    for collection, label in (
        (queue, "Review queue"),
        (not_queued, "Not-queued findings"),
    ):
        if any(
            not isinstance(item, str) or not item
            for item in collection
        ):
            raise ValueError(f"{label} contains an invalid finding ID.")

        if len(collection) != len(set(collection)):
            raise ValueError(f"{label} contains duplicate finding IDs.")

    if queue != automatic_queue:
        raise ValueError(
            "Review queue does not match automatic triage."
        )

    expected_not_queued = [
        finding_id
        for finding_id in expected_ids
        if finding_id not in set(automatic_queue)
    ]

    if not_queued != expected_not_queued:
        raise ValueError(
            "Not-queued findings do not match automatic triage."
        )

    decision_ids = _validate_decision_rows(
        review.get("review_decisions"),
        set(automatic_queue),
    )

    if set(decision_ids) != set(automatic_queue):
        raise ValueError(
            "Review decisions must contain exactly the queued findings."
        )


def load_review(report, report_path):
    review_path = Path(report_path).with_name("human_review.json")

    if not review_path.exists():
        return build_initial_review(report, report_path), review_path

    with review_path.open("r", encoding="utf-8") as file:
        review = json.load(file)

    if not isinstance(review, dict):
        raise ValueError("Human review must be a JSON object.")

    if Path(review.get("source_report", "")).resolve() != Path(
        report_path
    ).resolve():
        raise ValueError("Human review references a different source report.")

    validate_review(report, review)
    return review, review_path


def save_review(report, review_path, decisions, review_queue=None):
    review_path = Path(review_path)
    finding_ids = _finding_ids(report)
    automatic_queue = get_automatic_review_queue(report)

    if review_queue is not None and list(review_queue) != automatic_queue:
        raise ValueError(
            "A manually supplied queue cannot override automatic triage."
        )

    queue_set = set(automatic_queue)

    review = {
        "schema_version": "2.0",
        "source_report": str(
            review_path.with_name("detections.json").resolve()
        ),
        "review_queue": automatic_queue,
        "review_decisions": decisions,
        "not_queued_finding_ids": [
            finding_id
            for finding_id in finding_ids
            if finding_id not in queue_set
        ],
    }

    validate_review(report, review)

    temporary_path = review_path.with_suffix(
        review_path.suffix + ".tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(review, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(review_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return review
