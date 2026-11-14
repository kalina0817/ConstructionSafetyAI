import unittest
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from associate_worker_findings import find_source_detection
from build_adjudicated_report import build_adjudicated_report
from evaluation_metrics import binary_metrics, cohen_kappa
from evaluate_safety_system import (
    evaluate,
    load_gold_annotations,
    load_system_predictions,
)
from ppe_safety_rules import RECOMMENDATION_RULES
from reasoning_engine import (
    build_scene_graph,
    evaluate_safety_rules,
    generate_review_candidates,
)
from safety_reasoning import build_worker_recommendations


class AdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_folder = Path(self.temp_dir.name)
        self.detections_file = self.run_folder / "detections.json"
        self.candidate = {
            "candidate_type": "ppe_prediction_review",
            "class_name": "no_goggle",
            "detector": "PPE",
            "confidence": 0.8,
            "bbox_xyxy": [1, 2, 10, 12],
        }
        self.detections_report = {
            "review_analysis": {"candidates": [self.candidate]}
        }
        self.review_report = {
            "source_report": str(self.detections_file.resolve()),
            "review_decisions": [{
                "candidate_index": 0,
                "candidate": self.candidate,
                "decision": "Confirmed",
                "reviewer_notes": "Visible in the image",
            }],
        }

    def test_adjudication_preserves_detection_box(self):
        report = build_adjudicated_report(
            self.detections_report,
            self.review_report,
            self.run_folder,
        )

        self.assertEqual(report["summary"]["confirmed_predictions"], 1)
        self.assertEqual(
            report["confirmed_findings"][0]["bbox_xyxy"],
            [1, 2, 10, 12],
        )

    def test_rejects_review_for_different_candidate(self):
        self.review_report["review_decisions"][0]["candidate"] = {
            **self.candidate,
            "confidence": 0.1,
        }

        with self.assertRaisesRegex(ValueError, "does not match"):
            build_adjudicated_report(
                self.detections_report,
                self.review_report,
                self.run_folder,
            )

    def test_rejects_incomplete_review(self):
        self.review_report["review_decisions"] = []

        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_adjudicated_report(
                self.detections_report,
                self.review_report,
                self.run_folder,
            )

    def test_rejects_invalid_decision(self):
        self.review_report["review_decisions"][0]["decision"] = "Maybe"

        with self.assertRaisesRegex(ValueError, "Invalid decision"):
            build_adjudicated_report(
                self.detections_report,
                self.review_report,
                self.run_folder,
            )

    def test_accepts_empty_review_when_there_are_no_candidates(self):
        report = build_adjudicated_report(
            {"review_analysis": {"candidates": []}},
            {
                "source_report": str(self.detections_file.resolve()),
                "review_decisions": [],
            },
            self.run_folder,
        )

        self.assertEqual(report["total_reviewed_candidates"], 0)
        self.assertEqual(report["summary"]["pending_predictions"], 0)


class DetectionLinkageTests(unittest.TestCase):
    def test_source_detection_matches_reviewed_box(self):
        first = {
            "detector": "PPE",
            "class_name": "no_vest",
            "confidence": 0.8,
            "bbox_xyxy": [1, 2, 10, 12],
        }
        second = {**first, "bbox_xyxy": [20, 22, 30, 32]}
        finding = {**first}

        self.assertIs(
            find_source_detection(finding, [first, second]),
            first,
        )

    def test_ambiguous_legacy_match_is_not_guessed(self):
        first = {
            "detector": "PPE",
            "class_name": "no_vest",
            "confidence": 0.8,
            "bbox_xyxy": [1, 2, 10, 12],
        }
        second = {**first, "bbox_xyxy": [20, 22, 30, 32]}
        finding = {
            "detector": "PPE",
            "class_name": "no_vest",
            "confidence": 0.8,
        }

        self.assertIsNone(find_source_detection(finding, [first, second]))

    def test_spatial_candidates_keep_detection_provenance(self):
        detections = [
            {
                "detector": "PPE",
                "class_name": "Person",
                "confidence": 0.9,
                "bbox_xyxy": [10, 10, 30, 50],
            },
            {
                "detector": "Vehicles",
                "class_name": "Roller",
                "confidence": 0.8,
                "bbox_xyxy": [35, 10, 70, 50],
            },
        ]
        graph = build_scene_graph(detections, 100, 100)
        review = generate_review_candidates(graph)
        candidate = review["candidates"][0]

        self.assertEqual(candidate["person_analysis_index"], 0)
        self.assertEqual(candidate["machinery_analysis_index"], 1)
        self.assertEqual(candidate["person_bbox_xyxy"], [10, 10, 30, 50])
        self.assertEqual(candidate["machinery_bbox_xyxy"], [35, 10, 70, 50])
        self.assertEqual(candidate["machinery_detector"], "Vehicles")
        self.assertAlmostEqual(
            candidate["normalized_edge_distance"],
            5 / (100 ** 2 + 100 ** 2) ** 0.5,
            places=4,
        )

    def test_scene_graph_keeps_containment_distinct_from_wearing(self):
        graph = build_scene_graph(
            [
                {
                    "detector": "PPE",
                    "class_name": "Person",
                    "confidence": 0.9,
                    "bbox_xyxy": [0, 0, 100, 100],
                },
                {
                    "detector": "PPE",
                    "class_name": "helmet",
                    "confidence": 0.8,
                    "bbox_xyxy": [10, 5, 30, 20],
                },
            ],
            100,
            100,
        )

        self.assertEqual(len(graph["relationships"]), 1)
        self.assertEqual(
            graph["relationships"][0]["predicate"],
            "inside_person_bbox",
        )
        self.assertIn("does not establish", graph["relationships"][0][
            "interpretation_limit"
        ])

    def test_distant_machinery_does_not_create_proximity_candidate(self):
        graph = build_scene_graph(
            [
                {
                    "detector": "PPE",
                    "class_name": "Person",
                    "confidence": 0.9,
                    "bbox_xyxy": [0, 0, 20, 40],
                },
                {
                    "detector": "Vehicles",
                    "class_name": "Excavator",
                    "confidence": 0.8,
                    "bbox_xyxy": [90, 90, 100, 100],
                },
            ],
            100,
            100,
        )

        self.assertEqual(generate_review_candidates(graph)["candidate_count"], 0)

    def test_screening_findings_are_uncertain_and_never_risk_scored(self):
        graph = build_scene_graph(
            [
                {
                    "detector": "PPE",
                    "class_name": "Person",
                    "confidence": 0.9,
                    "bbox_xyxy": [0, 0, 80, 100],
                },
                {
                    "detector": "PPE",
                    "class_name": "no_vest",
                    "confidence": 0.75,
                    "bbox_xyxy": [10, 10, 70, 90],
                },
                {
                    "detector": "Vehicles",
                    "class_name": "Roller",
                    "confidence": 0.8,
                    "bbox_xyxy": [75, 0, 100, 100],
                },
            ],
            100,
            100,
        )
        assessment = evaluate_safety_rules(graph)

        self.assertEqual(
            assessment["summary"]["potential_missing_ppe"],
            1,
        )
        self.assertEqual(
            assessment["summary"]["potential_worker_machinery_interaction"],
            1,
        )
        self.assertIsNone(assessment["risk_assessment"]["risk_score"])
        self.assertTrue(all(
            finding["status"] == "requires_human_review"
            and finding["uncertainty_reasons"]
            for finding in assessment["findings"]
        ))

    def test_binary_metrics_and_rater_agreement(self):
        metrics = binary_metrics([1, 0, 1, 0], [1, 1, 0, 0])

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(
            cohen_kappa([1, 1, 0, 0], [1, 1, 0, 1]),
            0.5,
        )

    def test_every_missing_ppe_class_has_a_cautious_recommendation(self):
        classes = {
            "no_boots",
            "no_gloves",
            "no_goggle",
            "no_helmet",
            "no_vest",
        }
        self.assertEqual(set(RECOMMENDATION_RULES), classes)

        findings = [
            {"ppe_category": class_name, "human_decision": "Confirmed"}
            for class_name in sorted(classes)
        ]
        recommendations = build_worker_recommendations(findings)

        self.assertEqual(len(recommendations), len(classes))
        self.assertTrue(all(
            item["recommendation_status"] == "recommendation_generated"
            and item["recommendation"]
            and item["limitation"]
            for item in recommendations
        ))


class IntegratedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_evaluates_only_supported_categories_and_reports_metrics(self):
        annotations = {
            ("scene.jpg", "potential_missing_ppe"): 1,
            ("scene.jpg", "potential_worker_machinery_interaction"): 0,
        }
        predictions = {
            "scene.jpg": {"potential_missing_ppe"},
        }

        report = evaluate(annotations, predictions)

        self.assertEqual(report["overall"]["true_positive"], 1)
        self.assertEqual(report["overall"]["true_negative"], 1)
        self.assertEqual(
            report["categories"]["potential_missing_ppe"]["recall"],
            1.0,
        )

    def test_uncertain_annotations_are_excluded_but_counted(self):
        annotations = {
            ("scene.jpg", "potential_missing_ppe"): 1,
            ("scene.jpg", "potential_worker_machinery_interaction"): None,
        }
        predictions = {
            "scene.jpg": {
                "potential_missing_ppe",
                "potential_worker_machinery_interaction",
            },
        }

        report = evaluate(annotations, predictions)

        self.assertEqual(report["overall"]["support"], 1)
        self.assertEqual(report["annotation_summary"]["total_pairs"], 2)
        self.assertEqual(report["annotation_summary"]["evaluated_pairs"], 1)
        self.assertEqual(
            report["annotation_summary"]["excluded_uncertain_pairs"],
            1,
        )
        self.assertEqual(
            report["categories"][
                "potential_worker_machinery_interaction"
            ]["support"],
            0,
        )
        self.assertIsNone(report["categories"][
            "potential_worker_machinery_interaction"
        ]["accuracy"])

    def test_all_uncertain_annotations_produce_no_performance_scores(self):
        report = evaluate(
            {
                ("scene.jpg", "potential_missing_ppe"): None,
            },
            {"scene.jpg": set()},
        )

        self.assertEqual(report["overall"]["support"], 0)
        self.assertIsNone(report["overall"]["accuracy"])
        self.assertEqual(
            report["annotation_summary"]["excluded_uncertain_pairs"],
            1,
        )

    def test_gold_csv_accepts_uncertain_and_excludes_uncertain_rater_pairs(self):
        annotation_path = self.root / "annotations.csv"
        with annotation_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "image_id",
                "finding_category",
                "ground_truth",
                "rater_1",
                "rater_2",
            ])
            writer.writerow([
                "scene.jpg",
                "potential_missing_ppe",
                "unassessable",
                "uncertain",
                "1",
            ])

        annotations, raters = load_gold_annotations(annotation_path)

        self.assertIsNone(annotations[("scene.jpg", "potential_missing_ppe")])
        self.assertEqual(raters["rater_1"], [])
        self.assertEqual(raters["rater_2"], [])
        self.assertEqual(raters["excluded_uncertain_pairs"], 1)

    def test_gold_csv_rejects_partial_rater_columns(self):
        annotation_path = self.root / "annotations.csv"
        with annotation_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "image_id",
                "finding_category",
                "ground_truth",
                "rater_1",
            ])
            writer.writerow([
                "scene.jpg",
                "potential_missing_ppe",
                "1",
                "1",
            ])

        with self.assertRaisesRegex(ValueError, "both rater_1 and rater_2"):
            load_gold_annotations(annotation_path)

    def test_system_report_coverage_must_be_complete(self):
        report_path = self.root / "run" / "detections.json"
        report_path.parent.mkdir()
        with report_path.open("w", encoding="utf-8") as file:
            json.dump({
                "input_image": "scene.jpg",
                "safety_assessment": {"findings": []},
            }, file)

        with self.assertRaisesRegex(ValueError, "labeled images"):
            load_system_predictions(self.root, {"scene.jpg", "other.jpg"})


if __name__ == "__main__":
    unittest.main()
