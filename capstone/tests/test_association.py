
import unittest

from construction_safety.association import (
    extract_object_associations,
    extract_relationships,
)


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

    # Cross-detector object association tests

    def _excavator_pair(self):
        vehicles = {
            "object_id": "object_0000",
            "detector": "Vehicles",
            "class_name": "Excavator",
            "object_category": "machinery",
            "bbox_xyxy": [324, 177, 591, 394],
        }
        toolset = {
            "object_id": "object_0001",
            "detector": "Toolset-35",
            "class_name": "Excavator",
            "object_category": "other",
            "bbox_xyxy": [307, 174, 593, 395],
        }
        return vehicles, toolset

    def test_matching_excavators_from_different_detectors_are_associated(self):
        vehicles, toolset = self._excavator_pair()

        associations = extract_object_associations(
            [vehicles, toolset]
        )

        self.assertEqual(len(associations), 1)
        self.assertEqual(
            associations[0]["predicate"],
            "possible_same_physical_object",
        )

    def test_association_contains_both_object_ids_and_iou(self):
        vehicles, toolset = self._excavator_pair()

        associations = extract_object_associations(
            [vehicles, toolset]
        )

        association = associations[0]

        self.assertEqual(
            association["member_ids"],
            ["object_0000", "object_0001"],
        )
        self.assertAlmostEqual(
            association["evidence"]["intersection_over_union"],
            0.91667,
            places=5,
        )
        self.assertEqual(
            association["evidence"]["threshold"],
            0.50,
        )

    def test_same_detector_predictions_are_not_associated(self):
        vehicles, toolset = self._excavator_pair()
        toolset["detector"] = "Vehicles"

        associations = extract_object_associations(
            [vehicles, toolset]
        )

        self.assertEqual(associations, [])

    def test_different_class_names_are_not_associated(self):
        vehicles, toolset = self._excavator_pair()
        toolset["class_name"] = "Bulldozer"

        associations = extract_object_associations(
            [vehicles, toolset]
        )

        self.assertEqual(associations, [])

    def test_non_overlapping_boxes_are_not_associated(self):
        vehicles, toolset = self._excavator_pair()
        toolset["bbox_xyxy"] = [10, 10, 50, 50]

        associations = extract_object_associations(
            [vehicles, toolset]
        )

        self.assertEqual(associations, [])

    def test_toolset_excavator_can_match_vehicle_excavator(self):
        vehicles, toolset = self._excavator_pair()

        self.assertEqual(vehicles["object_category"], "machinery")
        self.assertEqual(toolset["object_category"], "other")

        associations = extract_object_associations(
            [vehicles, toolset]
        )

        self.assertEqual(len(associations), 1)

    def test_association_does_not_modify_original_nodes(self):
        vehicles, toolset = self._excavator_pair()
        original_vehicles = vehicles.copy()
        original_toolset = toolset.copy()

        extract_object_associations([vehicles, toolset])

        self.assertEqual(vehicles, original_vehicles)
        self.assertEqual(toolset, original_toolset)


if __name__ == "__main__":
    unittest.main()
