import unittest
import inspect
from pathlib import Path

from streamlit.testing.v1 import AppTest

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
        self.assertTrue(any(
            "Context-aware vision reasoning" in item.value
            for item in app.title
        ))

    def test_scenario_evaluation_page_renders_without_labels(self):
        app = AppTest.from_file(
            str(PROJECT_ROOT / "pages" / "evaluation.py"),
            default_timeout=45,
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(any(
            "Independent scenario evaluation" in item.value
            for item in app.title
        ))

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
        self.assertEqual(len(sha256_file(PROJECT_ROOT / "README.md")), 64)


if __name__ == "__main__":
    unittest.main()
