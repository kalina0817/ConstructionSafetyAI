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
        self.assertIn("does not establish", relation["interpretation_limit"])

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

    def test_all_findings_require_review_and_do_not_score_risk(self):
        graph = build_scene_graph(
            [
                {
                    "detector": "PPE",
                    "class_name": "no_vest",
                    "confidence": 0.7,
                    "bbox_xyxy": [10, 10, 50, 60],
                },
            ],
            100,
            100,
        )
        assessment = assess_safety(graph)

        self.assertEqual(
            assessment["findings"][0]["status"],
            "requires_human_review",
        )
        self.assertTrue(assessment["findings"][0]["uncertainty_reasons"])
        self.assertIsNone(assessment["risk_assessment"]["risk_score"])
        self.assertIsNone(assessment["risk_assessment"]["risk_probability"])

    def test_invalid_detection_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid bounding box"):
            build_scene_graph(
                [{
                    "detector": "PPE",
                    "class_name": "Person",
                    "bbox_xyxy": [0, 0, 0, 10],
                }],
                100,
                100,
            )


if __name__ == "__main__":
    unittest.main()
