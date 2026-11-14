import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from evaluate_detector_baselines import run_evaluation, sha256_file


ROOT = Path(__file__).resolve().parents[1]


class CapstoneApplicationTests(unittest.TestCase):
    def test_navigation_opens_research_overview(self):
        app = AppTest.from_file(
            str(ROOT / "app.py"),
            default_timeout=45,
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(any(
            "Context-aware vision reasoning" in title.value
            for title in app.title
        ))

    def test_research_methods_page_renders_without_annotations(self):
        app = AppTest.from_file(
            str(ROOT / "research_workbench.py"),
            default_timeout=45,
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(any(
            "Research methods and evaluation" in title.value
            for title in app.title
        ))

    def test_locked_test_evaluation_requires_explicit_confirmation(self):
        with self.assertRaisesRegex(ValueError, "test split is locked"):
            run_evaluation("ppe", split="test")

    def test_checkpoint_hash_is_sha256(self):
        test_file = ROOT / "tests" / "test_capstone_app.py"

        self.assertEqual(len(sha256_file(test_file)), 64)


if __name__ == "__main__":
    unittest.main()
