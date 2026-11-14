
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from construction_safety.inference import run_inference


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Vector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class InferenceThresholdTests(unittest.TestCase):
    def test_detector_specific_threshold_is_used_and_recorded(self):
        box = SimpleNamespace(
            cls=[Scalar(8)],
            xyxy=[Vector([10, 10, 50, 50])],
            conf=[Scalar(0.42)],
        )
        result = SimpleNamespace(
            boxes=[box],
            names={8: "Mobile Crane"},
        )
        predict = Mock(return_value=[result])
        model = SimpleNamespace(predict=predict)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "scene.png"
            Image.new("RGB", (100, 100)).save(image_path)
            with patch(
                "construction_safety.inference.YOLO",
                return_value=model,
            ) as detector:
                _, report = run_inference(
                    image_path,
                    confidence=0.25,
                    model_names=("vehicles",),
                    confidence_by_model={"vehicles": 0.40},
                )

        self.assertTrue(detector.call_args.args[0].endswith(".pt"))
        self.assertEqual(predict.call_args.kwargs["conf"], 0.40)
        self.assertEqual(
            report["configuration"]["confidence_thresholds"],
            {"vehicles": 0.40},
        )
        self.assertEqual(
            report["detections"][0]["confidence_threshold"],
            0.40,
        )

    def test_roller_threshold_remains_070_when_vehicle_slider_is_higher(self):
        boxes = [
            SimpleNamespace(
                cls=[Scalar(9)],
                xyxy=[Vector([10, 10, 40, 40])],
                conf=[Scalar(0.69)],
            ),
            SimpleNamespace(
                cls=[Scalar(9)],
                xyxy=[Vector([45, 10, 75, 40])],
                conf=[Scalar(0.72)],
            ),
            SimpleNamespace(
                cls=[Scalar(8)],
                xyxy=[Vector([10, 45, 40, 75])],
                conf=[Scalar(0.79)],
            ),
            SimpleNamespace(
                cls=[Scalar(8)],
                xyxy=[Vector([45, 45, 75, 75])],
                conf=[Scalar(0.82)],
            ),
        ]
        result = SimpleNamespace(
            boxes=boxes,
            names={8: "Mobile Crane", 9: "Roller"},
        )
        predict = Mock(return_value=[result])
        model = SimpleNamespace(predict=predict)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "scene.png"
            Image.new("RGB", (100, 100)).save(image_path)

            with patch(
                "construction_safety.inference.YOLO",
                return_value=model,
            ):
                _, report = run_inference(
                    image_path,
                    confidence=0.25,
                    model_names=("vehicles",),
                    confidence_by_model={"vehicles": 0.80},
                    confidence_by_class={"vehicles": {"Roller": 0.70}},
                )

        self.assertEqual(predict.call_args.kwargs["conf"], 0.70)
        self.assertEqual(
            [item["class_name"] for item in report["detections"]],
            ["Roller", "Mobile Crane"],
        )
        self.assertEqual(
            [item["confidence_threshold"] for item in report["detections"]],
            [0.70, 0.80],
        )

    def test_roller_threshold_does_not_raise_other_vehicle_thresholds(self):
        boxes = [
            SimpleNamespace(
                cls=[Scalar(9)],
                xyxy=[Vector([10, 10, 40, 40])],
                conf=[Scalar(0.65)],
            ),
            SimpleNamespace(
                cls=[Scalar(8)],
                xyxy=[Vector([45, 10, 75, 40])],
                conf=[Scalar(0.42)],
            ),
        ]
        result = SimpleNamespace(
            boxes=boxes,
            names={8: "Mobile Crane", 9: "Roller"},
        )
        predict = Mock(return_value=[result])
        model = SimpleNamespace(predict=predict)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "scene.png"
            Image.new("RGB", (100, 100)).save(image_path)

            with patch(
                "construction_safety.inference.YOLO",
                return_value=model,
            ):
                _, report = run_inference(
                    image_path,
                    confidence=0.25,
                    model_names=("vehicles",),
                    confidence_by_model={"vehicles": 0.40},
                    confidence_by_class={"vehicles": {"Roller": 0.70}},
                )

        self.assertEqual(predict.call_args.kwargs["conf"], 0.40)
        self.assertEqual(
            [item["class_name"] for item in report["detections"]],
            ["Mobile Crane"],
        )
        self.assertEqual(
            report["detections"][0]["confidence_threshold"],
            0.40,
        )

    def test_invalid_detector_specific_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            run_inference(
                "unused.png",
                confidence_by_model={"vehicles": 1.1},
            )

    def test_unknown_detector_threshold_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "configured detector names"):
            run_inference(
                "unused.png",
                confidence_by_model={"unknown": 0.4},
            )


if __name__ == "__main__":
    unittest.main()
