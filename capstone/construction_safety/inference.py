
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

from construction_safety.config import MODEL_SPECS, PROJECT_ROOT
from construction_safety.reasoning import assess_safety, build_scene_graph


def _sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def run_inference(
    image_path,
    confidence=0.25,
    image_size=640,
    model_names=("ppe", "toolset", "vehicles"),
    confidence_by_model=None,
    confidence_by_class=None,
):
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")

    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("Image size must be a positive integer.")

    if not model_names or any(
        name not in MODEL_SPECS for name in model_names
    ):
        raise ValueError("Select one or more configured detector models.")

    confidence_by_model = confidence_by_model or {}

    if any(name not in MODEL_SPECS for name in confidence_by_model):
        raise ValueError(
            "Confidence overrides must use configured detector names."
        )

    if any(
        not isinstance(value, (int, float)) or not 0 <= value <= 1
        for value in confidence_by_model.values()
    ):
        raise ValueError(
            "Detector confidence thresholds must be between 0 and 1."
        )

    confidence_by_class = confidence_by_class or {}

    if any(name not in MODEL_SPECS for name in confidence_by_class):
        raise ValueError(
            "Class confidence overrides must use configured detector names."
        )

    for class_thresholds in confidence_by_class.values():
        if not isinstance(class_thresholds, dict):
            raise ValueError(
                "Class confidence overrides must be dictionaries."
            )

        if any(
            not isinstance(value, (int, float)) or not 0 <= value <= 1
            for value in class_thresholds.values()
        ):
            raise ValueError(
                "Class confidence thresholds must be between 0 and 1."
            )

    image_path = Path(image_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    try:
        with Image.open(image_path) as source:
            source.verify()

        with Image.open(image_path) as source:
            width, height = source.size

    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not read the selected image: {error}"
        ) from error

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    detections = []
    checkpoints = {}
    nms_thresholds = {}

    annotated_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(annotated_image)

    detector_colors = {
        "PPE": (31, 107, 84),
        "Toolset": (31, 99, 164),
        "Vehicles": (192, 110, 31),
        "Toolset-35": (128, 76, 160),
    }

    for model_key in model_names:
        spec = MODEL_SPECS[model_key]

        model_confidence = confidence_by_model.get(
            model_key,
            confidence,
        )

        class_thresholds = confidence_by_class.get(
            model_key,
            {},
        )

        prediction_confidence = min(
            [model_confidence, *class_thresholds.values()]
        )

        # Use stricter NMS for Vehicles to reduce overlapping
        # same-class duplicate detections.
        # Other detectors retain the existing 0.70 setting.
        nms_iou_threshold = (
            0.50 if model_key == "vehicles" else 0.70
        )

        nms_thresholds[model_key] = nms_iou_threshold

        checkpoint = PROJECT_ROOT.parent / spec["checkpoint"]

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{spec['label']} checkpoint not found: {checkpoint}"
            )

        checkpoints[model_key] = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        }

        model = YOLO(str(checkpoint))

        result = model.predict(
            source=str(image_path),
            conf=prediction_confidence,
            imgsz=image_size,
            iou=nms_iou_threshold,
            device="cpu",
            verbose=False,
        )[0]

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            detection_confidence = float(box.conf[0].item())

            effective_threshold = class_thresholds.get(
                class_name,
                model_confidence,
            )

            if detection_confidence < effective_threshold:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            detection = {
                "detection_id": f"detection_{len(detections):05d}",
                "detector": spec["label"],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(detection_confidence, 5),
                "confidence_threshold": effective_threshold,
                "bbox_xyxy": [x1, y1, x2, y2],
            }

            detections.append(detection)

            color = detector_colors[spec["label"]]

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color,
                width=3,
            )

            draw.text(
                (x1, max(0, y1 - 14)),
                (
                    f"{spec['label']}: "
                    f"{class_name} {detection_confidence:.2f}"
                ),
                fill="white",
                stroke_width=2,
                stroke_fill=color,
            )

    scene_graph = build_scene_graph(
        detections,
        width,
        height,
    )

    assessment = assess_safety(scene_graph)

    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_image": image_path.name,
        "image_dimensions": {
            "width": width,
            "height": height,
        },
        "configuration": {
            "confidence_threshold": confidence,
            "confidence_thresholds": {
                name: confidence_by_model.get(name, confidence)
                for name in model_names
            },
            "class_confidence_thresholds": {
                name: thresholds
                for name, thresholds in confidence_by_class.items()
            },
            "nms_iou_thresholds": nms_thresholds,
            "image_size": image_size,
            "device": "cpu",
            "detectors": list(model_names),
        },
        "checkpoint_provenance": checkpoints,
        "detections": detections,
        "scene_graph": scene_graph,
        "safety_assessment": assessment,
        "interpretation_limit": (
            "Detector outputs are predictions. Relationships and findings "
            "require human review and do not establish physical site "
            "conditions."
        ),
    }

    return annotated_image, report


def save_run(report, output_root=None):
    output_root = Path(
        output_root or PROJECT_ROOT / "reports" / "inference_runs"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    run_folder = output_root / timestamp

    run_folder.mkdir(
        parents=False,
        exist_ok=False,
    )

    report_path = run_folder / "detections.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return run_folder, report_path
