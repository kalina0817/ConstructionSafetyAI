from copy import deepcopy

from construction_safety.inference import run_inference
from construction_safety.pipeline_config import (
    get_preset,
    validate_pipeline_config,
)
from construction_safety.registry import get_method


def _build_execution_record(config):
    """Describe the actual implementation used by each pipeline stage."""
    perception_methods = {
        detector: {
            "method_id": method_id,
            "implementation": get_method(method_id)["implementation"],
            "execution_mode": get_method(method_id)["execution_mode"],
            "status": "executed",
        }
        for detector, method_id in config["perception"].items()
    }

    stages = {}

    for stage in (
        "association",
        "spatial_reasoning",
        "scene_understanding",
        "hazard_screening",
        "explanation",
        "recommendations",
    ):
        method_id = config[stage]
        specification = get_method(method_id)

        if specification["execution_mode"] == "disabled":
            status = "disabled"
        elif stage in ("explanation", "recommendations"):
            status = "executed_integrated"
        elif specification["execution_mode"] == "integrated":
            status = "executed_integrated"
        else:
            status = "executed"

        stages[stage] = {
            "method_id": method_id,
            "implementation": specification["implementation"],
            "execution_mode": specification["execution_mode"],
            "status": status,
            "version": specification["version"],
        }

    return {
        "perception": perception_methods,
        "stages": stages,
        "limitations": [
            (
                "Association is independently extracted; spatial reasoning "
                "still uses the existing scene-graph construction."
            ),
            (
                "Hazard screening, explanations, and recommendations "
                "currently share the existing safety-assessment implementation."
            ),
            (
                "Scene understanding is disabled in the baseline pipeline."
            ),
        ],
    }


def run_pipeline(
    image_path,
    config=None,
    confidence=0.25,
    image_size=640,
    confidence_by_model=None,
):
    """
    Execute a validated pipeline configuration using existing implementations.

    Existing inference and report fields are preserved. Pipeline metadata is
    added as a separate section.
    """
    if config is None:
        config = get_preset("baseline")
    else:
        config = deepcopy(config)

    validate_pipeline_config(config)

    detector_order = ("ppe", "toolset", "vehicles", "toolset35")
    model_names = tuple(
        detector
        for detector in detector_order
        if detector in config["perception"]
    )

    annotated_image, report = run_inference(
        image_path=image_path,
        confidence=confidence,
        image_size=image_size,
        model_names=model_names,
        confidence_by_model=confidence_by_model,
    )

    report["pipeline"] = {
        "schema_version": "1.0",
        "preset": config["preset"],
        "configuration": deepcopy(config),
        "execution": _build_execution_record(config),
    }

    return annotated_image, report
