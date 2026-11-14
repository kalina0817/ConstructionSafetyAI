
from copy import deepcopy

from construction_safety.registry import get_method, validate_method


REQUIRED_PERCEPTION_DETECTORS = {"ppe", "toolset", "vehicles"}
OPTIONAL_PERCEPTION_DETECTORS = {"toolset35"}

PERCEPTION_DETECTORS = (
    REQUIRED_PERCEPTION_DETECTORS | OPTIONAL_PERCEPTION_DETECTORS
)

REVIEW_POLICIES = {"selective", "all_findings"}

BASELINE_CONFIG = {
    "preset": "baseline",
    "perception": {
        "ppe": "ppe_yolo11n_existing",
        "toolset": "toolset_yolo11n_existing",
        "vehicles": "vehicles_yolo11n_existing",
    },
    "association": "existing_geometric_relationships",
    "spatial_reasoning": "existing_geometric_scene_graph",
    "scene_understanding": "disabled_scene_understanding",
    "hazard_screening": "existing_rule_based_screening",
    "explanation": "existing_evidence_based_explanation",
    "recommendations": "existing_rule_based_recommendations",
    "human_review": "selective",
}

PRESETS = {
    "baseline": {
        "label": "Baseline",
        "status": "available",
        "description": (
            "Existing detectors, geometric scene graph, and rule-based "
            "screening."
        ),
        "config": BASELINE_CONFIG,
    },
    "advanced": {
        "label": "Advanced",
        "status": "planned",
        "description": (
            "Reserved for additional implemented and experimentally "
            "validated methods."
        ),
        "config": None,
    },
    "hybrid": {
        "label": "Hybrid",
        "status": "planned",
        "description": (
            "Reserved for a verified combination of learned methods "
            "and explicit rules."
        ),
        "config": None,
    },
}


def get_preset(preset_name):
    """Return a copy of an executable preset configuration."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown pipeline preset: {preset_name!r}")

    preset = PRESETS[preset_name]

    if preset["status"] != "available" or preset["config"] is None:
        raise ValueError(
            f"Pipeline preset {preset_name!r} is not implemented yet."
        )

    return deepcopy(preset["config"])


def get_preset_catalog():
    """Return preset descriptions without exposing mutable configurations."""
    return {
        name: {
            "label": item["label"],
            "status": item["status"],
            "description": item["description"],
        }
        for name, item in PRESETS.items()
    }


def validate_pipeline_config(config):
    """Validate every configured stage and its selected method."""
    if not isinstance(config, dict):
        raise ValueError("Pipeline configuration must be a dictionary.")

    required_stages = {
        "preset",
        "perception",
        "association",
        "spatial_reasoning",
        "scene_understanding",
        "hazard_screening",
        "explanation",
        "recommendations",
        "human_review",
    }

    missing = required_stages - config.keys()
    extra = config.keys() - required_stages

    if missing:
        raise ValueError(
            f"Pipeline configuration is missing: {sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"Pipeline configuration has unknown fields: {sorted(extra)}"
        )

    perception = config["perception"]

    if not isinstance(perception, dict):
        raise ValueError("Perception configuration must be a dictionary.")

    configured_detectors = set(perception)

    missing_detectors = (
        REQUIRED_PERCEPTION_DETECTORS - configured_detectors
    )
    unknown_detectors = configured_detectors - PERCEPTION_DETECTORS

    if missing_detectors:
        raise ValueError(
            "Perception is missing required detectors: "
            f"{sorted(missing_detectors)}"
        )

    if unknown_detectors:
        raise ValueError(
            "Perception contains unsupported detectors: "
            f"{sorted(unknown_detectors)}"
        )

    for detector, method_id in perception.items():
        specification = validate_method(
            method_id,
            expected_stage="perception",
        )
        if specification.get("model_name") != detector:
            raise ValueError(
                f"Method {method_id!r} is not the registered "
                f"{detector!r} detector."
            )

    stage_fields = {
        "association": "association",
        "spatial_reasoning": "spatial_reasoning",
        "scene_understanding": "scene_understanding",
        "hazard_screening": "hazard_screening",
        "explanation": "explanation",
        "recommendations": "recommendations",
    }

    for field, expected_stage in stage_fields.items():
        validate_method(
            config[field],
            expected_stage=expected_stage,
        )

    if config["human_review"] not in REVIEW_POLICIES:
        raise ValueError(
            f"Unsupported human-review policy: {config['human_review']!r}"
        )

    if config["preset"] not in PRESETS:
        raise ValueError(f"Unknown pipeline preset: {config['preset']!r}")

    return True


def update_stage(config, stage, method_id, detector=None):
    """
    Return a validated copy with exactly one stage selection changed.

    For perception, provide detector='ppe', 'toolset', 'vehicles',
    or the optional 'toolset35'.
    Other stages use their configuration field name.
    """
    updated = deepcopy(config)

    if stage == "perception":
        if detector not in PERCEPTION_DETECTORS:
            raise ValueError(
                "Perception updates require detector='ppe', 'toolset', "
                "'vehicles', or 'toolset35'."
            )

        specification = validate_method(
            method_id,
            expected_stage="perception",
        )

        if specification.get("model_name") != detector:
            raise ValueError(
                f"Method {method_id!r} is not the registered "
                f"{detector!r} detector."
            )

        updated["perception"][detector] = method_id

    elif stage == "human_review":
        if method_id not in REVIEW_POLICIES:
            raise ValueError(
                f"Unsupported human-review policy: {method_id!r}"
            )
        updated["human_review"] = method_id

    else:
        if detector is not None:
            raise ValueError(
                "The detector argument is only valid for perception."
            )

        stage_fields = {
            "association": "association",
            "spatial_reasoning": "spatial_reasoning",
            "scene_understanding": "scene_understanding",
            "hazard_screening": "hazard_screening",
            "explanation": "explanation",
            "recommendations": "recommendations",
        }

        if stage not in stage_fields:
            raise ValueError(f"Unknown pipeline stage: {stage!r}")

        validate_method(method_id, expected_stage=stage)
        updated[stage_fields[stage]] = method_id

    validate_pipeline_config(updated)
    return updated
