import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from construction_safety.evaluation import (
    binary_metrics,
    cohen_kappa,
    evaluate,
    load_annotations,
    load_predictions,
)


class ScenarioEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_metrics_and_kappa(self):
        metrics = binary_metrics([1, 0, 1, 0], [1, 1, 0, 0])

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(cohen_kappa([1, 1, 0, 0], [1, 1, 0, 1]), 0.5)

    def test_uncertain_annotations_are_excluded_and_counted(self):
        report = evaluate(
            {
                ("scene.jpg", "potential_missing_ppe"): 1,
                ("scene.jpg", "potential_worker_machinery_interaction"): None,
            },
            {"scene.jpg": {
                "potential_missing_ppe",
                "potential_worker_machinery_interaction",
            }},
        )

        self.assertEqual(report["overall"]["support"], 1)
        self.assertEqual(report["annotation_summary"]["total_pairs"], 2)
        self.assertEqual(report["annotation_summary"]["evaluated_pairs"], 1)
        self.assertEqual(
            report["annotation_summary"]["excluded_uncertain_pairs"],
            1,
        )

    def test_all_uncertain_has_undefined_metrics(self):
        report = evaluate(
            {("scene.jpg", "potential_missing_ppe"): None},
            {"scene.jpg": set()},
        )

        self.assertEqual(report["overall"]["support"], 0)
        self.assertIsNone(report["overall"]["accuracy"])

    def test_annotations_preserve_uncertain_rater_counts(self):
        path = self.root / "annotations.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
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

        annotations, raters = load_annotations(path)

        self.assertIsNone(
            annotations[("scene.jpg", "potential_missing_ppe")]
        )
        self.assertEqual(raters["excluded_uncertain_pairs"], 1)
        self.assertEqual(raters["rater_1"], [])

    def test_duplicate_or_unsupported_gold_rows_are_rejected(self):
        path = self.root / "duplicate.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image_id", "finding_category", "ground_truth"])
            writer.writerow(["scene.jpg", "potential_missing_ppe", "0"])
            writer.writerow(["scene.jpg", "potential_missing_ppe", "1"])

        with self.assertRaisesRegex(ValueError, "Duplicate annotation"):
            load_annotations(path)

    def test_missing_current_report_is_an_explicit_error(self):
        report = self.root / "run" / "detections.json"
        report.parent.mkdir()
        report.write_text(
            json.dumps({
                "input_image": "scene.jpg",
                "safety_assessment": {"findings": []},
            }),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "No compatible report"):
            load_predictions(self.root, {"scene.jpg", "missing.jpg"})


if __name__ == "__main__":
    unittest.main()
