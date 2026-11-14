import unittest

from construction_safety.association import extract_relationships


class AssociationTests(unittest.TestCase):

    def setUp(self):
        self.person = {
            "object_id": "object_0000",
            "object_category": "person",
            "bbox_xyxy": [10, 10, 100, 100],
        }
        self.helmet = {
            "object_id": "object_0001",
            "object_category": "ppe_prediction",
            "bbox_xyxy": [20, 15, 40, 30],
        }
        self.machinery = {
            "object_id": "object_0002",
            "object_category": "machinery",
            "bbox_xyxy": [105, 10, 150, 100],
        }

    def test_ppe_containment_relationship(self):
        relationships = extract_relationships(
            [self.person, self.helmet],
            200,
            200,
        )

        self.assertEqual(len(relationships), 1)
        relationship = relationships[0]
        self.assertEqual(relationship["predicate"], "inside_person_bbox")
        self.assertEqual(relationship["source_id"], "object_0001")
        self.assertEqual(relationship["target_id"], "object_0000")
        self.assertIn("does not establish", relationship["interpretation_limit"])

    def test_ppe_containment_does_not_claim_ppe_is_worn(self):
        relationships = extract_relationships(
            [self.person, self.helmet],
            200,
            200,
        )
        self.assertIn(
            "does not establish",
            relationships[0]["interpretation_limit"],
        )

    def test_machinery_proximity_relationship(self):
        relationships = extract_relationships(
            [self.person, self.machinery],
            200,
            200,
        )

        self.assertEqual(len(relationships), 1)
        relationship = relationships[0]
        self.assertEqual(
            relationship["predicate"],
            "near_machinery_in_image",
        )
        self.assertEqual(relationship["source_id"], "object_0000")
        self.assertEqual(relationship["target_id"], "object_0002")
        self.assertIn("normalized_edge_distance", relationship["evidence"])

    def test_distant_machinery_is_not_associated(self):
        distant = {
            **self.machinery,
            "bbox_xyxy": [190, 190, 200, 200],
        }
        relationships = extract_relationships(
            [self.person, distant],
            200,
            200,
        )
        self.assertEqual(relationships, [])

    def test_missing_ppe_prediction_can_be_associated(self):
        missing_helmet = {
            **self.helmet,
            "object_category": "missing_ppe_prediction",
        }
        relationships = extract_relationships(
            [self.person, missing_helmet],
            200,
            200,
        )
        self.assertEqual(len(relationships), 1)
        self.assertEqual(
            relationships[0]["predicate"],
            "inside_person_bbox",
        )

    def test_relationship_ids_are_unique_and_ordered(self):
        relationships = extract_relationships(
            [self.person, self.helmet, self.machinery],
            200,
            200,
        )
        ids = [item["relationship_id"] for item in relationships]
        self.assertEqual(
            ids,
            [f"relationship_{index:04d}" for index in range(len(ids))],
        )

    def test_invalid_image_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_relationships([self.person], 0, 200)

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_relationships(
                [self.person],
                200,
                200,
                machinery_distance_ratio=1.5,
            )

        with self.assertRaises(ValueError):
            extract_relationships(
                [self.person],
                200,
                200,
                containment_threshold=-0.1,
            )

    def test_invalid_node_geometry_is_rejected(self):
        invalid = {
            **self.person,
            "bbox_xyxy": [10, 10, 10, 100],
        }
        with self.assertRaisesRegex(ValueError, "invalid bounding box"):
            extract_relationships([invalid], 200, 200)

    def test_missing_node_identity_is_rejected(self):
        invalid = {
            "object_category": "person",
            "bbox_xyxy": [10, 10, 100, 100],
        }
        with self.assertRaisesRegex(ValueError, "object_id"):
            extract_relationships([invalid], 200, 200)


if __name__ == "__main__":
    unittest.main()
