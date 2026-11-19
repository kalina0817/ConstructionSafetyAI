
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

REPORT_ROOT = PROJECT_ROOT / "reports"
INFERENCE_ROOT = REPORT_ROOT / "inference_runs"
EVALUATION_ROOT = REPORT_ROOT / "detector_evaluations"


MODEL_SPECS = {
    "ppe": {
        "label": "PPE",
        "checkpoint": Path(
            "runs/detect/runs/detect/"
            "ppe_yolo11n_postcorrection_15ep/weights/best.pt"
        ),
        "dataset": Path(
            "data/standardized/construction-ppe/data.yaml"
        ),
        "training_args": Path(
            "runs/detect/runs/detect/"
            "ppe_yolo11n_postcorrection_15ep/args.yaml"
        ),
    },
    "toolset": {
        "label": "Toolset",
        "checkpoint": Path(
            "runs/detect/runs/detect/"
            "toolset_yolo11n_continuation_final5-2/weights/best.pt"
        ),
        "dataset": Path(
            "data/standardized/"
            "construction-toolset-safety-9class/data.yaml"
        ),
        "training_args": Path(
            "runs/detect/runs/detect/"
            "toolset_yolo11n_continuation_final5-2/args.yaml"
        ),
    },
    "vehicles": {
        "label": "Vehicles",
        "checkpoint": Path(
            "runs/detect/runs/detect/"
            "vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt"
        ),
        "dataset": Path(
            "data/standardized/construction-vehicle/data.yaml"
        ),
        "training_args": Path(
            "runs/detect/runs/detect/"
            "vehicle_yolo11n_corrected_dataset_5ep/args.yaml"
        ),
    },
    "toolset35": {
        "label": "Toolset-35",
        "checkpoint": Path(
            "models/toolset35_final_model/"
            "toolset35_yolo11n_best.pt"
        ),
        "dataset": Path(
            "data/standardized/"
            "construction-toolset-35class/data.yaml"
        ),
        "training_metadata": Path(
            "models/toolset35_final_model/training_metadata.json"
        ),
    },
}


# Model alternatives evaluated in the GPU and local CPU benchmarks.
# Paths are relative to WORKSPACE_ROOT.
MODEL_VARIANTS = {
    "ppe": {
        "yolo11n": {
            "architecture": "yolo",
            "checkpoint": Path(
                "runs/detect/runs/detect/"
                "ppe_yolo11n_postcorrection_15ep/weights/best.pt"
            ),
        },
        "rtdetr_l": {
            "architecture": "rtdetr",
            "checkpoint": Path(
                "models/rtdetr_l_comparison/"
                "ConstructionSafetyAI_model_comparison/"
                "rtdetr_l_ppe_best.pt"
            ),
        },
    },
    "toolset35": {
        "yolo11n": {
            "architecture": "yolo",
            "checkpoint": Path(
                "models/toolset35_final_model/"
                "toolset35_yolo11n_best.pt"
            ),
        },
        "rtdetr_l": {
            "architecture": "rtdetr",
            "checkpoint": Path(
                "models/rtdetr_l_comparison/"
                "ConstructionSafetyAI_model_comparison/"
                "rtdetr_l_toolset35_best.pt"
            ),
        },
    },
    "vehicles": {
        "yolo11n": {
            "architecture": "yolo",
            "checkpoint": Path(
                "runs/detect/runs/detect/"
                "vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt"
            ),
        },
        "rtdetr_l": {
            "architecture": "rtdetr",
            "checkpoint": Path(
                "models/rtdetr_l_comparison/"
                "ConstructionSafetyAI_model_comparison/"
                "rtdetr_l_vehicles_best.pt"
            ),
        },
    },
    "toolset": {
        "yolo11n": {
            "architecture": "yolo",
            "checkpoint": Path(
                "runs/detect/runs/detect/"
                "toolset_yolo11n_continuation_final5-2/weights/best.pt"
            ),
        },
    },
}


MACHINERY_CLASSES = {
    "boringmachine",
    "bulldozer",
    "dump truck",
    "excavator",
    "forklift",
    "grader",
    "loader",
    "mixer truck",
    "mobile crane",
    "roller",
}


SCREENING_CATEGORIES = {
    "potential_missing_ppe",
    "potential_worker_machinery_interaction",
}


# Image-space proximity thresholds.
# These are screening heuristics, not physical-distance measurements.
MACHINERY_REVIEW_DISTANCE_RATIO = 0.05
TOOL_REVIEW_DISTANCE_RATIO = 0.08

# Minimum fraction of a PPE prediction's bounding box
# that must overlap the person's bounding box.
PPE_PERSON_CONTAINMENT_THRESHOLD = 0.80
