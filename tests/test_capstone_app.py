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
            "Construction Safety AI" in title.value
            for title in app.title
        ))

    def test_research_methods_page_renders_without_annotations(self):
        app = AppTest.from_file(
            str(ROOT / "research_workbench.py"),
            default_timeout=45,
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(any(
            "Research and testing" in title.value
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



def test_switching_models_keeps_saved_review_attached_to_its_run(tmp_path):
    import json

    report = {"detection_model": "YOLO11n", "reasoning_model": "Scene-graph rules",
              "reasoning_status": "COMPLETED", "reasoning_error": None, "worker_ppe": [],
              "review_analysis": {"candidates": [{"candidate_type": "missing_ppe_review"}]}}
    (tmp_path / "screening_report.json").write_text(json.dumps(report), encoding="utf-8")
    app = AppTest.from_file(str(ROOT / "integrated_app.py"), default_timeout=45).run()
    app.session_state["run_folder"] = str(tmp_path)
    app.selectbox(key="reasoning_model").select("Qwen3-VL").run()
    assert not app.exception
    next(b for b in app.button if "save" in b.label.lower() and "review" in b.label.lower()).click().run()
    assert not app.exception
    saved = json.loads((tmp_path / "human_review.json").read_text(encoding="utf-8"))
    assert Path(saved["source_report"]) == tmp_path / "detections.json"
