
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import scripts.evaluate_detector as detector_evaluator
from construction_safety.config import MODEL_SPECS, WORKSPACE_ROOT
from scripts.evaluate_detector import run_evaluation, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CapstoneAppTests(unittest.TestCase):
    def test_app_routes_to_capstone_overview(self):
        app = AppTest.from_file(
            str(PROJECT_ROOT / "app.py"),
            default_timeout=45,
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
               "ConstructionSafetyAI" in item.value
                for item in app.title
            )
        )

    def test_scenario_evaluation_page_renders_without_labels(self):
        app = AppTest.from_file(
            str(PROJECT_ROOT / "pages" / "evaluation.py"),
            default_timeout=45,
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Independent scenario evaluation" in item.value
                for item in app.title
            )
        )

    def test_screening_page_has_separate_vehicle_threshold(self):
        app = AppTest.from_file(
            str(PROJECT_ROOT / "pages" / "screening.py"),
            default_timeout=45,
        ).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.slider[0].value, 0.25)
        self.assertEqual(app.slider[1].value, 0.40)

    def test_locked_test_split_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(ValueError, "test split is locked"):
            run_evaluation("ppe", split="test")

    def test_validation_defaults_to_unlocked_split(self):
        default_split = inspect.signature(run_evaluation).parameters[
            "split"
        ].default

        self.assertEqual(default_split, "val")

        checkpoint = WORKSPACE_ROOT / MODEL_SPECS["ppe"]["checkpoint"]
        self.assertTrue(checkpoint.is_file())

    def test_sha256_is_recorded_as_fixed_length_digest(self):
        digest = sha256_file(PROJECT_ROOT / "README.md")

        self.assertEqual(len(digest), 64)

    def test_toolset35_uses_training_metadata_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            checkpoint = root / "toolset35.pt"
            dataset = root / "toolset35.yaml"
            metadata = root / "training_metadata.json"
            evaluations = root / "evaluations"

            checkpoint.write_bytes(b"mock checkpoint")

            metadata.write_text(
                json.dumps(
                    {
                        "model": "YOLO11n",
                        "epochs": 50,
                    }
                ),
                encoding="utf-8",
            )

            dataset.write_text(
                "test: images/test\n"
                "names:\n"
                "  0: Person\n"
                "  1: hammer\n",
                encoding="utf-8",
            )

            fake_result = SimpleNamespace(
                box=SimpleNamespace(maps=[0.50, 0.25]),
                results_dict={
                    "metrics/precision(B)": 0.70,
                },
            )

            fake_model = SimpleNamespace(
                names={
                    0: "Person",
                    1: "hammer",
                },
                val=lambda **kwargs: fake_result,
            )

            toolset35_spec = {
                "label": "Toolset-35",
                "checkpoint": Path("toolset35.pt"),
                "dataset": Path("toolset35.yaml"),
                "training_metadata": Path("training_metadata.json"),
            }

            with (
                patch.dict(
                    detector_evaluator.MODEL_SPECS,
                    {"toolset35": toolset35_spec},
                ),
                patch.object(
                    detector_evaluator,
                    "WORKSPACE_ROOT",
                    root,
                ),
                patch.object(
                    detector_evaluator,
                    "EVALUATION_ROOT",
                    evaluations,
                ),
                patch.object(
                    detector_evaluator,
                    "YOLO",
                    return_value=fake_model,
                ) as mock_yolo,
            ):
                report_path, report = (
                    detector_evaluator.run_evaluation("toolset35")
                )

            artifacts = report["artifacts"]

            self.assertTrue(report_path.is_file())
            self.assertEqual(report["detector"], "toolset35")
            self.assertEqual(report["evaluation_split"], "val")
            self.assertFalse(report["test_split_was_locked"])

            self.assertEqual(
                artifacts["training_provenance_type"],
                "training_metadata",
            )
            self.assertEqual(
                artifacts["training_metadata"],
                str(metadata),
            )
            self.assertEqual(
                artifacts["training_metadata_sha256"],
                sha256_file(metadata),
            )
            self.assertEqual(
                artifacts["training_provenance_sha256"],
                sha256_file(metadata),
            )

            self.assertNotIn("training_args", artifacts)
            self.assertNotIn("training_args_sha256", artifacts)

            self.assertEqual(
                report["per_class_map50_95"],
                {
                    "Person": 0.50,
                    "hammer": 0.25,
                },
            )

            self.assertTrue(mock_yolo.called)
            self.assertEqual(
                mock_yolo.call_args.args[0],
                str(checkpoint),
            )

            saved_report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                saved_report["artifacts"]["training_provenance_type"],
                "training_metadata",
            )


if __name__ == "__main__":
    unittest.main()
