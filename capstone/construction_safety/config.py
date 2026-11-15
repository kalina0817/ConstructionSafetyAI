
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
            "data/processed/construction-ppe-corrected/data.yaml"
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
            "data/processed/construction-toolset-safety-clean/data.yaml"
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
            "data/processed/"
            "construction-vehicle-detection-deduplicated/data.yaml"
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

MACHINERY_REVIEW_DISTANCE_RATIO = 0.05
PPE_PERSON_CONTAINMENT_THRESHOLD = 0.80
