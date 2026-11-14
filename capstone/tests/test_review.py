import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from construction_safety.review import (
    build_initial_review,
    get_automatic_review_queue,
    load_review,
    save_review,
    validate_review,
)


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_dir = Path(self.temp_dir.name)
        self.report_path = self.run_dir / "detections.json"
        self.report = {
            "safety_assessment": {
                "findings": [
                    {
                        "finding_id": "finding_0000",
                        "review_disposition": "review_required",
                    },
                    {
                        "finding_id": "finding_0001",
                        "review_disposition": "insufficient_evidence",
                    },
                ]
            }
        }
        self.report_path.write_text(
            json.dumps(self.report),
            encoding="utf-8",
        )

    def test_initial_review_uses_automatic_queue(self):
        review = build_initial_review(self.report, self.report_path)

        self.assertEqual(review["schema_version"], "2.0")
        self.assertEqual(
            review["review_queue"],
            ["finding_0000", "finding_0001"],
        )
        self.assertEqual(
            [row["decision"] for row in review["review_decisions"]],
            ["Pending", "Pending"],
        )
        self.assertEqual(review["not_queued_finding_ids"], [])

    def test_no_review_required_findings_are_excluded(self):
        report = {
            "safety_assessment": {
                "findings": [
                    {
                        "finding_id": "finding_0000",
                        "review_disposition": "no_review_required",
                    },
                    {
                        "finding_id": "finding_0001",
                        "review_disposition": "review_required",
                    },
                ]
            }
        }

        queue = get_automatic_review_queue(report)

        self.assertEqual(queue, ["finding_0001"])

    def test_missing_disposition_is_queued_conservatively(self):
        report = {
            "safety_assessment": {
                "findings": [
                    {"finding_id": "finding_0000"},
                ]
            }
        }

        self.assertEqual(
            get_automatic_review_queue(report),
            ["finding_0000"],
        )

    def test_unknown_disposition_is_queued_conservatively(self):
        report = {
            "safety_assessment": {
                "findings": [
                    {
                        "finding_id": "finding_0000",
                        "review_disposition": "unknown_value",
                    },
                ]
            }
        }

        self.assertEqual(
            get_automatic_review_queue(report),
            ["finding_0000"],
        )

    def test_saves_and_reloads_automatic_review_decisions(self):
        report_before = self.report_path.read_text(encoding="utf-8")
        _, review_path = load_review(self.report, self.report_path)

        decisions = [
            {
                "finding_id": "finding_0000",
                "decision": "Confirmed",
                "reviewer_notes": "Visible evidence verified.",
            },
            {
                "finding_id": "finding_0001",
                "decision": "Uncertain",
                "reviewer_notes": "Image evidence is ambiguous.",
            },
        ]

        saved = save_review(self.report, review_path, decisions)
        loaded, _ = load_review(self.report, self.report_path)

        self.assertEqual(saved["schema_version"], "2.0")
        self.assertEqual(
            loaded["review_queue"],
            ["finding_0000", "finding_0001"],
        )
        self.assertEqual(loaded["review_decisions"], decisions)
        self.assertEqual(loaded["not_queued_finding_ids"], [])
        self.assertEqual(
            self.report_path.read_text(encoding="utf-8"),
            report_before,
        )

    def test_empty_queue_is_valid_when_all_findings_need_no_review(self):
        report = {
            "safety_assessment": {
                "findings": [
                    {
                        "finding_id": "finding_0000",
                        "review_disposition": "no_review_required",
                    },
                    {
                        "finding_id": "finding_0001",
                        "review_disposition": "no_review_required",
                    },
                ]
            }
        }
        _, review_path = load_review(report, self.report_path)

        saved = save_review(report, review_path, [])

        self.assertEqual(saved["review_queue"], [])
        self.assertEqual(saved["review_decisions"], [])
        self.assertEqual(
            saved["not_queued_finding_ids"],
            ["finding_0000", "finding_0001"],
        )

    def test_rejects_manual_queue_override(self):
        _, review_path = load_review(self.report, self.report_path)

        with self.assertRaisesRegex(
            ValueError,
            "cannot override automatic triage",
        ):
            save_review(
                self.report,
                review_path,
                [],
                review_queue=[],
            )

    def test_rejects_decision_for_finding_outside_queue(self):
        _, review_path = load_review(self.report, self.report_path)

        with self.assertRaisesRegex(ValueError, "unknown or unqueued"):
            save_review(
                self.report,
                review_path,
                [
                    {
                        "finding_id": "finding_9999",
                        "decision": "Confirmed",
                        "reviewer_notes": "",
                    },
                    {
                        "finding_id": "finding_0001",
                        "decision": "Uncertain",
                        "reviewer_notes": "",
                    },
                ],
            )

    def test_rejects_missing_decision_for_queued_finding(self):
        _, review_path = load_review(self.report, self.report_path)

        with self.assertRaisesRegex(ValueError, "exactly the queued"):
            save_review(
                self.report,
                review_path,
                [
                    {
                        "finding_id": "finding_0000",
                        "decision": "Pending",
                        "reviewer_notes": "",
                    },
                ],
            )

    def test_rejects_review_file_with_modified_queue(self):
        review = build_initial_review(self.report, self.report_path)
        review["review_queue"] = ["finding_0000"]

        with self.assertRaisesRegex(ValueError, "automatic triage"):
            validate_review(self.report, review)

    def test_rejects_inconsistent_not_queued_ids(self):
        review = build_initial_review(self.report, self.report_path)
        review["not_queued_finding_ids"] = ["finding_0000"]

        with self.assertRaisesRegex(ValueError, "automatic triage"):
            validate_review(self.report, review)

    def test_reads_legacy_v1_review(self):
        legacy_review = {
            "schema_version": "1.0",
            "source_report": str(self.report_path.resolve()),
            "review_decisions": [
                {
                    "finding_id": "finding_0000",
                    "decision": "Confirmed",
                    "reviewer_notes": "",
                },
                {
                    "finding_id": "finding_0001",
                    "decision": "Uncertain",
                    "reviewer_notes": "",
                },
            ],
        }

        review_path = self.run_dir / "human_review.json"
        review_path.write_text(
            json.dumps(legacy_review),
            encoding="utf-8",
        )

        loaded, _ = load_review(self.report, self.report_path)

        self.assertEqual(loaded["schema_version"], "1.0")
        self.assertEqual(
            loaded["review_decisions"],
            legacy_review["review_decisions"],
        )

    def test_rejects_incomplete_legacy_decisions(self):
        review = {
            "schema_version": "1.0",
            "source_report": str(self.report_path.resolve()),
            "review_decisions": [
                {
                    "finding_id": "finding_0000",
                    "decision": "Confirmed",
                    "reviewer_notes": "",
                },
            ],
        }

        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_review(self.report, review)

    def test_rejects_invalid_legacy_decisions(self):
        review = {
            "schema_version": "1.0",
            "source_report": str(self.report_path.resolve()),
            "review_decisions": [
                {
                    "finding_id": "finding_0000",
                    "decision": "Maybe",
                    "reviewer_notes": "",
                },
                {
                    "finding_id": "finding_0001",
                    "decision": "Pending",
                    "reviewer_notes": "",
                },
            ],
        }

        with self.assertRaisesRegex(ValueError, "Invalid review"):
            validate_review(self.report, review)


if __name__ == "__main__":
    unittest.main()
