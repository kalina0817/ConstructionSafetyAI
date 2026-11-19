import unittest
from copy import deepcopy

from construction_safety.pipeline_config import (
    get_preset,
    get_preset_catalog,
    update_stage,
    validate_pipeline_config,
)


class PipelineConfigTests(unittest.TestCase):

    def setUp(self):
        self.config = get_preset("baseline")

    def test_baseline_configuration_is_valid(self):
        self.assertTrue(validate_pipeline_config(self.config))

    def test_only_baseline_is_available(self):
        catalog = get_preset_catalog()
        self.assertEqual(catalog["baseline"]["status"], "available")
        self.assertEqual(catalog["advanced"]["status"], "planned")
        self.assertEqual(catalog["hybrid"]["status"], "planned")

    def test_planned_presets_cannot_be_loaded(self):
        for preset in ("advanced", "hybrid"):
            with self.subTest(preset=preset):
                with self.assertRaises(ValueError):
                    get_preset(preset)

    def test_preset_returns_independent_copy(self):
        first = get_preset("baseline")
        first["human_review"] = "all_findings"
        second = get_preset("baseline")
        self.assertEqual(second["human_review"], "selective")

    def test_detector_selection_changes_only_selected_detector(self):
        updated = update_stage(
            self.config,
            "perception",
            "ppe_yolo11n_existing",
            detector="ppe",
        )
        self.assertEqual(
            updated["perception"]["toolset35"],
            self.config["perception"]["toolset35"],
        )
        self.assertEqual(
            updated["perception"]["vehicles"],
            self.config["perception"]["vehicles"],
        )

    def test_detector_method_must_match_detector(self):
        with self.assertRaises(ValueError):
            update_stage(
                self.config,
                "perception",
                "toolset_yolo11n_existing",
                detector="ppe",
            )

    def test_stage_selection_changes_only_selected_stage(self):
        updated = update_stage(
            self.config,
            "spatial_reasoning",
            "existing_geometric_scene_graph",
        )
        for key in self.config:
            if key != "spatial_reasoning":
                self.assertEqual(updated[key], self.config[key])

    def test_original_configuration_is_not_modified(self):
        original = deepcopy(self.config)
        update_stage(
            self.config,
            "spatial_reasoning",
            "existing_geometric_scene_graph",
        )
        self.assertEqual(self.config, original)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            update_stage(
                self.config,
                "spatial_reasoning",
                "unknown_method",
            )

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            update_stage(self.config, "unknown_stage", "unknown_method")

    def test_invalid_review_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            update_stage(
                self.config,
                "human_review",
                "automatic_approval",
            )


if __name__ == "__main__":
    unittest.main()
