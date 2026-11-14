
METHOD_REGISTRY = {
    "ppe_yolo11n_existing": {
        "stage": "perception",
        "label": "Existing PPE YOLO11n",
        "implementation": "construction_safety.inference.run_inference",
        "model_name": "ppe",
        "availability": "available",
        "execution_mode": "existing",
        "version": "existing-checkpoint",
        "inputs": ["image"],
        "outputs": ["detections"],
        "limitations": [
            "Uses the existing trained PPE checkpoint.",
            "Detection does not establish whether PPE is correctly worn.",
        ],
    },
    "toolset_yolo11n_existing": {
        "stage": "perception",
        "label": "Existing Toolset YOLO11n",
        "implementation": "construction_safety.inference.run_inference",
        "model_name": "toolset",
        "availability": "available",
        "execution_mode": "existing",
        "version": "existing-checkpoint",
        "inputs": ["image"],
        "outputs": ["detections"],
        "limitations": [
            "Uses the existing trained toolset checkpoint.",
            "Object detection alone does not establish unsafe use.",
        ],
    },
    "vehicles_yolo11n_existing": {
        "stage": "perception",
        "label": "Existing Vehicle YOLO11n",
        "implementation": "construction_safety.inference.run_inference",
        "model_name": "vehicles",
        "availability": "available",
        "execution_mode": "existing",
        "version": "existing-checkpoint",
        "inputs": ["image"],
        "outputs": ["detections"],
        "limitations": [
            "Uses the existing vehicle checkpoint.",
            "Detection does not establish machine operation or danger.",
        ],
    },
    "toolset35_yolo11n_existing": {
        "stage": "perception",
        "label": "Existing Toolset-35 YOLO11n",
        "implementation": "construction_safety.inference.run_inference",
        "model_name": "toolset35",
        "availability": "available",
        "execution_mode": "existing",
        "version": "toolset35-final-checkpoint",
        "inputs": ["image"],
        "outputs": ["detections"],
        "limitations": [
            "Uses the existing 35-class Toolset checkpoint.",
            "General object detection does not establish unsafe activity.",
            "PPE detections do not establish that equipment is being worn.",
            "Excavator detections are not a substitute for the dedicated vehicle detector.",
        ],
    },
    "existing_geometric_relationships": {
        "stage": "association",
        "label": "Existing Geometric Relationships",
        "implementation": "construction_safety.association.extract_relationships",
        "availability": "available",
        "execution_mode": "existing",
        "version": "existing",
        "inputs": ["scene_nodes", "image_dimensions"],
        "outputs": ["relationships"],
        "limitations": [
            "Relationships are based on image-space geometry, not physical measurements.",
            "PPE containment does not establish that equipment is worn.",
        ],
    },
    "existing_geometric_scene_graph": {
        "stage": "spatial_reasoning",
        "label": "Existing Geometric Scene Graph",
        "implementation": "construction_safety.reasoning.build_scene_graph",
        "availability": "available",
        "execution_mode": "existing",
        "version": "existing",
        "inputs": ["detections", "image_dimensions"],
        "outputs": ["scene_graph"],
        "limitations": [
            "Uses the existing geometric scene-graph implementation.",
            "Relationships are image-based and require contextual interpretation.",
        ],
    },
    "disabled_scene_understanding": {
        "stage": "scene_understanding",
        "label": "Disabled",
        "implementation": None,
        "availability": "available",
        "execution_mode": "disabled",
        "version": "1.0",
        "inputs": ["image"],
        "outputs": [],
        "limitations": [],
    },
    "existing_rule_based_screening": {
        "stage": "hazard_screening",
        "label": "Existing Rule-Based Screening",
        "implementation": "construction_safety.hazard_screening.screen_hazards",
        "availability": "available",
        "execution_mode": "existing",
        "version": "existing",
        "inputs": ["scene_graph"],
        "outputs": ["findings"],
        "limitations": [
            "Implements the existing missing-PPE and worker-machinery rules.",
            "Findings are potential concerns, not confirmed hazards.",
        ],
    },
    "existing_evidence_based_explanation": {
        "stage": "explanation",
        "label": "Existing Evidence-Based Explanation",
        "implementation": "construction_safety.explanations.generate_explanations",
        "availability": "available",
        "execution_mode": "existing",
        "version": "existing",
        "inputs": ["findings"],
        "outputs": ["explanations"],
        "limitations": [
            "Explanations are rule-specific and depend on detector and image evidence.",
        ],
    },
    "existing_rule_based_recommendations": {
        "stage": "recommendations",
        "label": "Existing Rule-Based Recommendations",
        "implementation": "construction_safety.recommendations.generate_recommendations",
        "availability": "available",
        "execution_mode": "existing",
        "version": "existing",
        "inputs": ["findings"],
        "outputs": ["recommendations"],
        "limitations": [
            "Recommendations are rule-based and require verification against actual site conditions.",
        ],
    },
}


def get_method(method_id):
    """Return a registered method or raise an informative error."""
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError as error:
        raise ValueError(f"Unknown method ID: {method_id!r}") from error


def get_methods(stage=None, available_only=True):
    """Return registered methods, optionally filtered by stage."""
    methods = {}

    for method_id, specification in METHOD_REGISTRY.items():
        if stage is not None and specification["stage"] != stage:
            continue
        if available_only and specification["availability"] != "available":
            continue
        methods[method_id] = specification.copy()

    return methods


def validate_method(method_id, expected_stage=None):
    """Validate that a method exists and belongs to the expected stage."""
    specification = get_method(method_id)

    if specification["availability"] != "available":
        raise ValueError(
            f"Method {method_id!r} is not currently available."
        )

    if (
        expected_stage is not None
        and specification["stage"] != expected_stage
    ):
        raise ValueError(
            f"Method {method_id!r} belongs to stage "
            f"{specification['stage']!r}, not {expected_stage!r}."
        )

    return specification
