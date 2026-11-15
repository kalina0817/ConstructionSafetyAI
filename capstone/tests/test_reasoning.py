
import unittest

from construction_safety.reasoning import (
    assess_safety,
    build_scene_graph,
)


class SceneGraphTests(unittest.TestCase):
    def test_ppe_containment_is_not_claimed_as_wearing(self):
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
        relation = graph["relationships"][0]
        self.assertEqual(relation["predicate"], "inside_person_bbox")
        self.assertIn(
            "does not establish",
            relation["interpretation_limit"],
        )

    def test_machinery_relationship_is_thresholded_and_traceable(self):
        graph = build_scene_graph(
            [
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
            ],
            100,
            100,
        )

        relationship = graph["relationships"][0]
        self.assertEqual(
            relationship["predicate"],
            "near_machinery_in_image",
        )
        self.assertEqual(relationship["source_id"], "object_0000")
        self.assertEqual(relationship["target_id"], "object_0001")
        self.assertAlmostEqual(
            relationship["evidence"]["normalized_edge_distance"],
            5 / (100 ** 2 + 100 ** 2) ** 0.5,
            places=4,
        )

    def test_distant_machinery_does_not_trigger_finding(self):
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

        self.assertEqual(assess_safety(graph)["findings"], [])

    def test_missing_ppe_finding_is_automatic_and_threshold_qualified(self):
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
                    "class_name": "no_vest",
                    "confidence": 0.7,
                    "confidence_threshold": 0.25,
                    "bbox_xyxy": [10, 10, 50, 60],
                },
            ],
            100,
            100,
        )

        assessment = assess_safety(graph)

        self.assertEqual(len(assessment["findings"]), 1)
        finding = assessment["findings"][0]

        self.assertEqual(
            finding["finding_category"],
            "potential_missing_ppe",
        )
        self.assertEqual(finding["status"], "potential")
        self.assertEqual(
            finding["review_disposition"],
            "no_review_required",
        )
        self.assertTrue(finding["uncertainty_reasons"])
        self.assertEqual(
            finding["evidence"]["confidence_threshold"],
            0.25,
        )
        self.assertEqual(
            assessment["review_triage"]["review_queue"],
            [],
        )
        self.assertEqual(
            assessment["review_triage"]["disposition_counts"][
                "no_review_required"
            ],
            1,
        )
        self.assertIsNone(
            assessment["risk_assessment"]["risk_score"]
        )
        self.assertIsNone(
            assessment["risk_assessment"]["risk_probability"]
        )

    def test_missing_ppe_without_person_association_does_not_create_finding(self):
        graph = build_scene_graph(
            [
                {
                    "detector": "PPE",
                    "class_name": "no_vest",
                    "confidence": 0.9,
                    "confidence_threshold": 0.25,
                    "bbox_xyxy": [10, 10, 50, 60],
                },
            ],
            100,
            100,
        )

        self.assertEqual(assess_safety(graph)["findings"], [])

    def test_missing_ppe_below_confidence_threshold_does_not_create_finding(self):
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
                    "class_name": "no_vest",
                    "confidence": 0.2,
                    "confidence_threshold": 0.25,
                    "bbox_xyxy": [10, 10, 50, 60],
                },
            ],
            100,
            100,
        )

        self.assertEqual(assess_safety(graph)["findings"], [])

    def test_all_five_missing_ppe_classes_use_the_same_rule(self):
        missing_ppe_classes = [
            "no_boots",
            "no_gloves",
            "no_goggles",
            "no_helmet",
            "no_vest",
        ]

        detections = [
            {
                "detector": "PPE",
                "class_name": "Person",
                "confidence": 0.9,
                "bbox_xyxy": [0, 0, 100, 100],
            }
        ]

        for index, class_name in enumerate(missing_ppe_classes):
            detections.append(
                {
                    "detector": "PPE",
                    "class_name": class_name,
                    "confidence": 0.8,
                    "confidence_threshold": 0.25,
                    "bbox_xyxy": [
                        5 + index * 2,
                        5,
                        30 + index * 2,
                        30,
                    ],
                }
            )

        graph = build_scene_graph(detections, 100, 100)
        assessment = assess_safety(graph)

        self.assertEqual(len(assessment["findings"]), 5)
        self.assertEqual(
            {
                finding["evidence"]["class_name"]
                for finding in assessment["findings"]
            },
            set(missing_ppe_classes),
        )
        self.assertTrue(
            all(
                finding["review_disposition"] == "no_review_required"
                for finding in assessment["findings"]
            )
        )
        self.assertEqual(
            assessment["review_triage"]["review_queue"],
            [],
        )

    def test_worker_machinery_finding_is_automatic(self):
        graph = build_scene_graph(
            [
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
            ],
            100,
            100,
        )

        assessment = assess_safety(graph)

        self.assertEqual(len(assessment["findings"]), 1)
        finding = assessment["findings"][0]

        self.assertEqual(
            finding["finding_category"],
            "potential_worker_machinery_interaction",
        )
        self.assertEqual(finding["status"], "potential")
        self.assertEqual(
            finding["review_disposition"],
            "no_review_required",
        )
        self.assertEqual(
            assessment["review_triage"]["review_queue"],
            [],
        )

    def test_toolset35_ppe_classes_are_classified_as_ppe_predictions(self):
        ppe_classes = [
            "gloves",
            "safety glass",
            "safety headphones",
            "safety helmet",
            "safety mask",
            "safety_vest",
        ]

        graph = build_scene_graph(
            [
                {
                    "detector": "Toolset-35",
                    "class_name": class_name,
                    "confidence": 0.8,
                    "bbox_xyxy": [10, 10, 30, 30],
                }
                for class_name in ppe_classes
            ],
            100,
            100,
        )

        self.assertEqual(
            [node["object_category"] for node in graph["nodes"]],
            ["ppe_prediction"] * len(ppe_classes),
        )

    def test_toolset35_ppe_predictions_do_not_create_missing_ppe_findings(self):
        graph = build_scene_graph(
            [
                {
                    "detector": "PPE",
                    "class_name": "Person",
                    "confidence": 0.9,
                    "bbox_xyxy": [0, 0, 100, 100],
                },
                {
                    "detector": "Toolset-35",
                    "class_name": "safety helmet",
                    "confidence": 0.8,
                    "bbox_xyxy": [10, 5, 30, 20],
                },
            ],
            100,
            100,
        )

        self.assertEqual(
            graph["nodes"][1]["object_category"],
            "ppe_prediction",
        )
        self.assertEqual(
            assess_safety(graph)["findings"],
            [],
        )

    def test_toolset35_excavator_is_not_classified_as_machinery(self):
        graph = build_scene_graph(
            [
                {
                    "detector": "PPE",
                    "class_name": "Person",
                    "confidence": 0.9,
                    "bbox_xyxy": [10, 10, 30, 50],
                },
                {
                    "detector": "Toolset-35",
                    "class_name": "Excavator",
                    "confidence": 0.8,
                    "bbox_xyxy": [35, 10, 70, 50],
                },
            ],
            100,
            100,
        )

        self.assertEqual(graph["nodes"][1]["object_category"], "other")
        self.assertEqual(graph["relationships"], [])

    def test_vehicles_excavator_remains_machinery(self):
        graph = build_scene_graph(
            [
                {
                    "detector": "PPE",
                    "class_name": "Person",
                    "confidence": 0.9,
                    "bbox_xyxy": [10, 10, 30, 50],
                },
                {
                    "detector": "Vehicles",
                    "class_name": "Excavator",
                    "confidence": 0.8,
                    "bbox_xyxy": [35, 10, 70, 50],
                },
            ],
            100,
            100,
        )

        self.assertEqual(graph["nodes"][1]["object_category"], "machinery")
        self.assertEqual(
            graph["relationships"][0]["predicate"],
            "near_machinery_in_image",
        )

    def test_invalid_detection_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid bounding box"):
            build_scene_graph(
                [
                    {
                        "detector": "PPE",
                        "class_name": "Person",
                        "bbox_xyxy": [0, 0, 0, 10],
                    }
                ],
                100,
                100,
            )


if __name__ == "__main__":
    unittest.main()
