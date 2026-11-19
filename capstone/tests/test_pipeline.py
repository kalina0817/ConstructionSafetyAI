import unittest
from unittest.mock import patch
from PIL import Image

from construction_safety.pipeline import run_pipeline
from construction_safety.pipeline_config import get_preset, update_stage


class PipelineTests(unittest.TestCase):

    def setUp(self):
        self.config = get_preset("baseline")
        self.image = Image.new("RGB", (32, 32))
        self.report = {
            "schema_version": "1.0",
            "input_image": "sample.jpg",
            "image_dimensions": {"width": 32, "height": 32},
            "configuration": {"detectors": ["ppe", "toolset35", "vehicles"]},
            "checkpoint_provenance": {},
            "detections": [],
            "scene_graph": {"nodes": [], "relationships": []},
            "safety_assessment": {"findings": []},
            "interpretation_limit": "Existing interpretation limit.",
        }

    @patch("construction_safety.pipeline.run_inference")
    def test_baseline_pipeline_calls_existing_inference_once(self, mock_inference):
        mock_inference.return_value = (self.image, self.report.copy())

        image, report = run_pipeline("sample.jpg", self.config)

        mock_inference.assert_called_once()
        self.assertIs(image, self.image)
        self.assertIn("pipeline", report)

    @patch("construction_safety.pipeline.run_inference")
    def test_selected_detectors_are_passed_to_inference(self, mock_inference):
        mock_inference.return_value = (self.image, self.report.copy())

        run_pipeline("sample.jpg", self.config)

        kwargs = mock_inference.call_args.kwargs
        self.assertEqual(kwargs["model_names"], ("ppe", "toolset35", "vehicles"))
        self.assertEqual(
            kwargs["model_variants"],
            {
                "ppe": "yolo11n",
                "toolset35": "yolo11n",
                "vehicles": "yolo11n",
            },
        )

    @patch("construction_safety.pipeline.run_inference")
    def test_confidence_settings_are_preserved(self, mock_inference):
        mock_inference.return_value = (self.image, self.report.copy())
        overrides = {"vehicles": 0.40}

        run_pipeline(
            "sample.jpg",
            self.config,
            confidence=0.30,
            image_size=512,
            confidence_by_model=overrides,
        )

        kwargs = mock_inference.call_args.kwargs
        self.assertEqual(kwargs["confidence"], 0.30)
        self.assertEqual(kwargs["image_size"], 512)
        self.assertEqual(kwargs["confidence_by_model"], overrides)

    @patch("construction_safety.pipeline.run_inference")
    def test_existing_report_fields_are_preserved(self, mock_inference):
        original = self.report.copy()
        mock_inference.return_value = (self.image, original)

        _, report = run_pipeline("sample.jpg", self.config)

        for field, value in original.items():
            self.assertEqual(report[field], value)

    @patch("construction_safety.pipeline.run_inference")
    def test_pipeline_configuration_is_recorded(self, mock_inference):
        mock_inference.return_value = (self.image, self.report.copy())

        _, report = run_pipeline("sample.jpg", self.config)

        pipeline = report["pipeline"]
        self.assertEqual(pipeline["schema_version"], "1.0")
        self.assertEqual(pipeline["preset"], "baseline")
        self.assertEqual(pipeline["configuration"], self.config)

    @patch("construction_safety.pipeline.run_inference")
    def test_integrated_stages_are_identified(self, mock_inference):
        mock_inference.return_value = (self.image, self.report.copy())

        _, report = run_pipeline("sample.jpg", self.config)

        stages = report["pipeline"]["execution"]["stages"]
        self.assertEqual(
            stages["association"]["status"],
            "executed",
        )
        self.assertEqual(
            stages["hazard_screening"]["status"],
            "executed",
        )
        self.assertEqual(
            stages["explanation"]["status"],
            "executed_integrated",
        )
        self.assertEqual(
            stages["recommendations"]["status"],
            "executed_integrated",
        )
        self.assertEqual(
            stages["scene_understanding"]["status"],
            "disabled",
        )

    @patch("construction_safety.pipeline.run_inference")
    def test_detector_selection_remains_independent(self, mock_inference):
        mock_inference.return_value = (self.image, self.report.copy())

        updated = update_stage(
            self.config,
            "perception",
            "ppe_yolo11n_existing",
            detector="ppe",
        )

        run_pipeline("sample.jpg", updated)

        kwargs = mock_inference.call_args.kwargs
        self.assertEqual(kwargs["model_names"], ("ppe", "toolset35", "vehicles"))
        self.assertEqual(
            updated["perception"]["vehicles"],
            self.config["perception"]["vehicles"],
        )

    @patch("construction_safety.pipeline.run_inference")
    def test_invalid_configuration_stops_before_inference(self, mock_inference):
        invalid = self.config.copy()
        invalid["spatial_reasoning"] = "unknown_method"

        with self.assertRaises(ValueError):
            run_pipeline("sample.jpg", invalid)

        mock_inference.assert_not_called()

    @patch("construction_safety.pipeline.run_inference")
    def test_execution_records_all_three_detectors(self, mock_inference):
        mock_inference.return_value = (self.image, self.report.copy())

        _, report = run_pipeline("sample.jpg", self.config)

        perception = report["pipeline"]["execution"]["perception"]
        self.assertEqual(set(perception), {"ppe", "toolset35", "vehicles"})
        for detector in perception.values():
            self.assertEqual(detector["status"], "executed")


if __name__ == "__main__":
    unittest.main()
