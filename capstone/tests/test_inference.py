
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from construction_safety.inference import (
    _classify_screening_outcome,
    _deduplicate_detections,
    _suppress_same_class_ppe_duplicates,
    run_inference,
)


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


class ScreeningOutcomeTests(unittest.TestCase):
    def test_findings_produce_potential_outcome(self):
        result = _classify_screening_outcome(
            [{"finding_id": "finding_0001"}],
            ("ppe", "vehicles"),
        )

        self.assertEqual(result["outcome"], "POTENTIAL")
        self.assertEqual(result["finding_count"], 1)
        self.assertFalse(result["high_confidence_alert_available"])

    def test_complete_coverage_without_findings_is_no_supported(self):
        result = _classify_screening_outcome(
            [],
            ("ppe", "vehicles"),
        )

        self.assertEqual(result["outcome"], "NO_SUPPORTED")
        self.assertEqual(result["finding_count"], 0)
        self.assertFalse(result["high_confidence_alert_available"])
        self.assertIn("does not establish", result["explanation"])

    def test_incomplete_coverage_without_findings_is_insufficient(self):
        result = _classify_screening_outcome(
            [],
            ("ppe",),
        )

        self.assertEqual(result["outcome"], "INSUFFICIENT")
        self.assertFalse(result["high_confidence_alert_available"])

    def test_toolset_only_coverage_is_insufficient(self):
        result = _classify_screening_outcome(
            [],
            ("toolset",),
        )

        self.assertEqual(result["outcome"], "INSUFFICIENT")

    def test_high_confidence_detection_does_not_create_high_confidence_alert(self):
        result = _classify_screening_outcome(
            [{"finding_id": "finding_0001", "confidence": 0.99}],
            ("ppe", "vehicles"),
        )

        self.assertEqual(result["outcome"], "POTENTIAL")
        self.assertFalse(result["high_confidence_alert_available"])
        self.assertIn(
            "independently validated hazard-specific criterion",
            result["high_confidence_alert_limitation"],
        )


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

    def test_original_deduplication_associations_are_preserved(self):
        box = SimpleNamespace(
            cls=[Scalar(8)],
            xyxy=[Vector([10, 10, 50, 50])],
            conf=[Scalar(0.42)],
        )
        result = SimpleNamespace(
            boxes=[box],
            names={8: "Mobile Crane"},
        )
        model = SimpleNamespace(predict=Mock(return_value=[result]))

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
                )

        self.assertIn("deduplication_associations", report)
        self.assertEqual(report["deduplication_associations"], [])

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
                xyxy=[Vector([45, 10, 75, 75])],
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


class DeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.raw_detections = [
            {
                "detection_id": "detection_0000",
                "detector": "Vehicles",
                "class_name": "Excavator",
                "confidence": 0.82,
                "box_xyxy": [100, 100, 300, 300],
            },
            {
                "detection_id": "detection_0001",
                "detector": "Toolset-35",
                "class_name": "Excavator",
                "confidence": 0.94,
                "box_xyxy": [105, 105, 295, 295],
            },
            {
                "detection_id": "detection_0002",
                "detector": "PPE",
                "class_name": "helmet",
                "confidence": 0.91,
                "box_xyxy": [400, 100, 450, 160],
            },
        ]

        self.raw_scene_graph = {
            "nodes": [
                {
                    "object_id": "object_0000",
                    "detector": "Vehicles",
                    "class_name": "Excavator",
                    "confidence": 0.82,
                    "box_xyxy": [100, 100, 300, 300],
                },
                {
                    "object_id": "object_0001",
                    "detector": "Toolset-35",
                    "class_name": "Excavator",
                    "confidence": 0.94,
                    "box_xyxy": [105, 105, 295, 295],
                },
                {
                    "object_id": "object_0002",
                    "detector": "PPE",
                    "class_name": "helmet",
                    "confidence": 0.91,
                    "box_xyxy": [400, 100, 450, 160],
                },
            ],
            "object_associations": [
                {
                    "member_ids": ["object_0000", "object_0001"],
                    "predicate": "possible_same_physical_object",
                }
            ],
        }

    def test_highest_confidence_detection_is_retained(self):
        result = _deduplicate_detections(
            self.raw_detections,
            self.raw_scene_graph,
        )

        excavators = [
            item for item in result
            if item["class_name"] == "Excavator"
        ]

        self.assertEqual(len(excavators), 1)
        self.assertEqual(excavators[0]["detector"], "Toolset-35")
        self.assertEqual(excavators[0]["confidence"], 0.94)

    def test_duplicate_metadata_is_preserved(self):
        result = _deduplicate_detections(
            self.raw_detections,
            self.raw_scene_graph,
        )

        excavator = next(
            item for item in result
            if item["class_name"] == "Excavator"
        )

        self.assertEqual(excavator["possible_duplicate_count"], 2)
        self.assertEqual(
            excavator["associated_object_ids"],
            ["object_0000", "object_0001"],
        )

    def test_unassociated_detection_is_preserved(self):
        result = _deduplicate_detections(
            self.raw_detections,
            self.raw_scene_graph,
        )

        self.assertEqual(len(result), 2)

        helmets = [
            item for item in result
            if item["class_name"] == "helmet"
        ]

        self.assertEqual(len(helmets), 1)
        self.assertEqual(helmets[0]["confidence"], 0.91)

    def test_original_predictions_and_scene_graph_are_unchanged(self):
        original_detections = copy.deepcopy(self.raw_detections)
        original_scene_graph = copy.deepcopy(self.raw_scene_graph)

        _deduplicate_detections(
            self.raw_detections,
            self.raw_scene_graph,
        )

        self.assertEqual(self.raw_detections, original_detections)
        self.assertEqual(self.raw_scene_graph, original_scene_graph)


class SameClassPPESuppressionTests(unittest.TestCase):
    def make_detection(
        self,
        detection_id,
        class_name,
        confidence,
        bbox,
        detector="PPE",
    ):
        return {
            "detection_id": detection_id,
            "detector": detector,
            "class_name": class_name,
            "confidence": confidence,
            "box_xyxy": bbox,
        }

    def test_overlapping_no_helmet_predictions_keep_highest_confidence(self):
        detections = [
            self.make_detection(
                "no_helmet_1",
                "no_helmet",
                0.72,
                [0, 0, 100, 100],
            ),
            self.make_detection(
                "no_helmet_2",
                "no_helmet",
                0.91,
                [10, 10, 110, 110],
            ),
        ]

        result = _suppress_same_class_ppe_duplicates(detections)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["detection_id"], "no_helmet_2")
        self.assertEqual(result[0]["possible_duplicate_count"], 2)
        self.assertIn(
            "no_helmet_1",
            result[0]["suppressed_detection_ids"],
        )

    def test_different_ppe_classes_are_not_merged(self):
        detections = [
            self.make_detection(
                "missing_helmet",
                "no_helmet",
                0.90,
                [0, 0, 100, 100],
            ),
            self.make_detection(
                "helmet",
                "helmet",
                0.85,
                [0, 0, 100, 100],
            ),
            self.make_detection(
                "missing_vest",
                "no_vest",
                0.80,
                [0, 0, 100, 100],
            ),
        ]

        result = _suppress_same_class_ppe_duplicates(detections)

        self.assertEqual(len(result), 3)
        self.assertEqual(
            {item["detection_id"] for item in result},
            {"missing_helmet", "helmet", "missing_vest"},
        )

    def test_low_overlap_same_class_predictions_remain_separate(self):
        detections = [
            self.make_detection(
                "no_helmet_1",
                "no_helmet",
                0.90,
                [0, 0, 100, 100],
            ),
            self.make_detection(
                "no_helmet_2",
                "no_helmet",
                0.80,
                [60, 0, 160, 100],
            ),
        ]

        result = _suppress_same_class_ppe_duplicates(detections)

        self.assertEqual(len(result), 2)

    def test_suppression_does_not_modify_raw_detections(self):
        detections = [
            self.make_detection(
                "no_helmet_1",
                "no_helmet",
                0.70,
                [0, 0, 100, 100],
            ),
            self.make_detection(
                "no_helmet_2",
                "no_helmet",
                0.90,
                [10, 10, 110, 110],
            ),
        ]
        original = copy.deepcopy(detections)

        _suppress_same_class_ppe_duplicates(detections)

        self.assertEqual(detections, original)

    def test_different_detectors_are_not_suppressed_by_this_rule(self):
        detections = [
            self.make_detection(
                "ppe_no_helmet",
                "no_helmet",
                0.90,
                [0, 0, 100, 100],
                detector="PPE",
            ),
            self.make_detection(
                "other_no_helmet",
                "no_helmet",
                0.85,
                [0, 0, 100, 100],
                detector="Toolset",
            ),
        ]

        result = _suppress_same_class_ppe_duplicates(detections)

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
