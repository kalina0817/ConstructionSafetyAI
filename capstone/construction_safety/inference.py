
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import RTDETR, YOLO

from construction_safety.config import (
    MODEL_SPECS,
    MODEL_VARIANTS,
    PROJECT_ROOT,
)
from construction_safety.association import canonical_class_name
from construction_safety.reasoning import assess_safety, build_scene_graph


SAME_CLASS_DUPLICATE_IOU_THRESHOLD = 0.50

# Backward compatibility.
SAME_CLASS_PPE_DUPLICATE_IOU_THRESHOLD = (
    SAME_CLASS_DUPLICATE_IOU_THRESHOLD
)

PPE_CLASS_NAMES = {
    "boots",
    "gloves",
    "goggles",
    "helmet",
    "vest",
    "no_boots",
    "no_gloves",
    "no_goggles",
    "no_helmet",
    "no_vest",
    "safety glass",
    "safety glasses",
    "safety helmet",
    "safety vest",
    "safety mask",
    "safety headphones",
}


def _sha256(path):
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_iou_threshold(iou_threshold):
    """Validate a duplicate-suppression IoU threshold."""
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not math.isfinite(iou_threshold)
        or not 0 <= iou_threshold <= 1
    ):
        raise ValueError("IoU threshold must be between 0 and 1.")

    return float(iou_threshold)


def _confidence(detection):
    """Return a validated confidence value."""
    try:
        value = float(detection.get("confidence", 0))
    except (TypeError, ValueError):
        raise ValueError("Detection confidence must be numeric.")

    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Detection confidence must be between 0 and 1.")

    return value


def _find_root(parent, object_id):
    """Find the root of a union-find group."""
    while parent[object_id] != object_id:
        parent[object_id] = parent[parent[object_id]]
        object_id = parent[object_id]

    return object_id


def _union(parent, first_id, second_id):
    """Join two detection groups."""
    first_root = _find_root(parent, first_id)
    second_root = _find_root(parent, second_id)

    if first_root != second_root:
        parent[second_root] = first_root


def _intersection_over_union(box_a, box_b):
    """Calculate IoU between two box_xyxy bounding boxes."""
    if box_a is None or box_b is None:
        return 0.0

    try:
        if len(box_a) != 4 or len(box_b) != 4:
            raise ValueError(
                "Bounding boxes must contain four coordinates."
            )

        ax1, ay1, ax2, ay2 = map(float, box_a)
        bx1, by1, bx2, by2 = map(float, box_b)

    except (TypeError, ValueError):
        raise ValueError(
            "Bounding-box coordinates must contain four numeric values."
        )

    coordinates = (
        ax1, ay1, ax2, ay2,
        bx1, by1, bx2, by2,
    )

    if any(not math.isfinite(value) for value in coordinates):
        raise ValueError("Bounding-box coordinates must be finite.")

    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0

    intersection_width = max(
        0.0,
        min(ax2, bx2) - max(ax1, bx1),
    )
    intersection_height = max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )

    intersection = intersection_width * intersection_height
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _get_detection_for_node(node, detections):
    """Match a scene-graph node to its original detector prediction."""
    node_box = node.get("box_xyxy")

    if (
        not isinstance(node_box, (list, tuple))
        or len(node_box) != 4
    ):
        raise ValueError(
            "Scene-graph node has no valid box_xyxy: "
            f"{node.get('object_id')}"
        )

    node_box = list(node_box)
    detection_index = node.get("detection_index")

    if (
        isinstance(detection_index, int)
        and not isinstance(detection_index, bool)
        and 0 <= detection_index < len(detections)
    ):
        detection = detections[detection_index]

        if (
            detection.get("detector") == node.get("detector")
            and detection.get("class_name") == node.get("class_name")
            and detection.get("box_xyxy") == node_box
        ):
            return detection

    candidates = [
        detection
        for detection in detections
        if detection.get("detector") == node.get("detector")
        and detection.get("class_name") == node.get("class_name")
        and detection.get("box_xyxy") == node_box
    ]

    if not candidates:
        raise ValueError(
            "Could not match scene-graph node to a detection: "
            f"{node.get('object_id')}"
        )

    try:
        node_confidence = float(node.get("confidence") or 0)
    except (TypeError, ValueError):
        raise ValueError("Scene-graph node confidence must be numeric.")

    if not math.isfinite(node_confidence):
        raise ValueError("Scene-graph node confidence must be finite.")

    return min(
        candidates,
        key=lambda detection: abs(
            _confidence(detection) - node_confidence
        ),
    )


def _get_member_ids(group):
    """Collect unique scene-graph object IDs represented by a group."""
    member_ids = set()

    for detection in group:
        existing_ids = detection.get("associated_object_ids") or []

        if isinstance(existing_ids, (list, tuple, set)):
            member_ids.update(
                object_id
                for object_id in existing_ids
                if isinstance(object_id, str) and object_id
            )

    if member_ids:
        return sorted(member_ids)

    detection_ids = {
        detection["detection_id"]
        for detection in group
        if detection.get("detection_id")
    }

    return sorted(detection_ids)


def _add_duplicate_metadata(winner, group):
    """Attach duplicate-group and suppression metadata to the winner."""
    winner = dict(winner)

    member_ids = _get_member_ids(group)

    source_detection_ids = set()
    previously_suppressed_ids = set()

    for detection in group:
        detection_id = detection.get("detection_id")

        if detection_id:
            source_detection_ids.add(detection_id)

        suppressed = detection.get("suppressed_detection_ids") or []

        if isinstance(suppressed, (list, tuple, set)):
            previously_suppressed_ids.update(
                item for item in suppressed
                if isinstance(item, str) and item
            )

    winner_id = winner.get("detection_id")

    suppressed_ids = (
        source_detection_ids
        - ({winner_id} if winner_id else set())
    )

    suppressed_ids.update(previously_suppressed_ids)

    if winner_id:
        suppressed_ids.discard(winner_id)

    winner["possible_duplicate_count"] = len(member_ids)
    winner["associated_object_ids"] = member_ids
    winner["suppressed_detection_ids"] = sorted(suppressed_ids)

    return winner


def _merge_cross_detector_duplicates(
    raw_detections,
    raw_scene_graph,
):
    """
    Merge predictions explicitly associated with the same possible object.

    Cross-class merging requires a valid scene-graph association and
    compatible canonical class names.

    Original input detections are never modified.
    """
    if not raw_detections:
        return []

    if not isinstance(raw_detections, list):
        raise ValueError("Detections must be provided as a list.")

    if not isinstance(raw_scene_graph, dict):
        raise ValueError("Scene graph must be a dictionary.")

    detections = [
        dict(detection)
        for detection in raw_detections
    ]

    if any(not isinstance(detection, dict) for detection in raw_detections):
        raise ValueError("Every detection must be a dictionary.")

    nodes = raw_scene_graph.get("nodes", [])
    associations = raw_scene_graph.get("object_associations", [])

    if not isinstance(nodes, list):
        raise ValueError("Scene-graph nodes must be a list.")

    if not isinstance(associations, list):
        raise ValueError("Object associations must be a list.")

    detections_by_id = {}

    for detection in detections:
        detection_id = detection.get("detection_id")

        if (
            not isinstance(detection_id, str)
            or not detection_id.strip()
        ):
            raise ValueError(
                "Every detection must have a valid detection_id."
            )

        if detection_id in detections_by_id:
            raise ValueError(
                f"Duplicate detection_id: {detection_id}"
            )

        _confidence(detection)
        detections_by_id[detection_id] = detection

    parent = {
        detection_id: detection_id
        for detection_id in detections_by_id
    }

    node_to_detection_id = {}

    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError(
                "Every scene-graph node must be a dictionary."
            )

        object_id = node.get("object_id")

        if not isinstance(object_id, str) or not object_id.strip():
            continue

        if object_id in node_to_detection_id:
            raise ValueError(
                f"Duplicate scene-graph object_id: {object_id}"
            )

        detection = _get_detection_for_node(node, detections)
        detection_id = detection["detection_id"]

        associated_ids = detection.setdefault(
            "associated_object_ids",
            [],
        )

        if object_id not in associated_ids:
            associated_ids.append(object_id)

        node_to_detection_id[object_id] = detection_id

    for association in associations:
        if not isinstance(association, dict):
            continue

        if (
            association.get("predicate")
            != "possible_same_physical_object"
        ):
            continue

        member_ids = association.get("member_ids", [])

        if (
            not isinstance(member_ids, list)
            or len(member_ids) < 2
            or any(
                not isinstance(object_id, str)
                for object_id in member_ids
            )
        ):
            continue

        if any(
            object_id not in node_to_detection_id
            for object_id in member_ids
        ):
            continue

        member_detection_ids = list(dict.fromkeys(
            node_to_detection_id[object_id]
            for object_id in member_ids
        ))

        if len(member_detection_ids) < 2:
            continue

        canonical_names = [
            canonical_class_name(
                detections_by_id[detection_id].get("class_name")
            )
            for detection_id in member_detection_ids
        ]

        if (
            any(name is None for name in canonical_names)
            or len(set(canonical_names)) != 1
        ):
            continue

        first_id = member_detection_ids[0]

        for other_id in member_detection_ids[1:]:
            if first_id in parent and other_id in parent:
                _union(parent, first_id, other_id)

    groups = {}

    for detection_id, detection in detections_by_id.items():
        root = _find_root(parent, detection_id)
        groups.setdefault(root, []).append(detection)

    retained = []

    for group in groups.values():
        winner = max(
            group,
            key=_confidence,
        ).copy()

        retained.append(
            _add_duplicate_metadata(winner, group)
        )

    return retained


def _merge_same_class_duplicates(
    detections,
    iou_threshold=SAME_CLASS_DUPLICATE_IOU_THRESHOLD,
    same_detector_only=False,
):
    """
    Retain the highest-confidence prediction in each duplicate group.

    Exact class names must match. When same_detector_only is true,
    predictions from different detectors are never merged by this helper.
    """
    threshold = _validate_iou_threshold(iou_threshold)

    if not detections:
        return []

    ordered = sorted(
        enumerate(detections),
        key=lambda item: (-_confidence(item[1]), item[0]),
    )

    groups = []

    for _, detection in ordered:
        if "box_xyxy" not in detection:
            raise ValueError("Every detection must have box_xyxy.")

        class_name = str(
            detection.get("class_name", "")
        ).strip().casefold()

        if not class_name:
            raise ValueError("Every detection must have a class_name.")

        _confidence(detection)

        matching_group = None

        for group in groups:
            representative = group[0]

            representative_class = str(
                representative.get("class_name", "")
            ).strip().casefold()

            if representative_class != class_name:
                continue

            if (
                same_detector_only
                and representative.get("detector") != detection.get("detector")
            ):
                continue

            if any(
                (
                    not same_detector_only
                    or member.get("detector") == detection.get("detector")
                )
                and _intersection_over_union(
                    detection["box_xyxy"],
                    member["box_xyxy"],
                ) >= threshold
                for member in group
            ):
                matching_group = group
                break

        if matching_group is None:
            groups.append([detection])
        else:
            matching_group.append(detection)

    retained = []

    for group in groups:
        winner = max(
            group,
            key=_confidence,
        ).copy()

        retained.append(
            _add_duplicate_metadata(winner, group)
        )

    return retained


def _suppress_same_class_ppe_duplicates(
    detections,
    iou_threshold=SAME_CLASS_PPE_DUPLICATE_IOU_THRESHOLD,
):
    """
    Suppress same-class PPE duplicates within the same detector only.

    Predictions from different detectors remain available for the main
    cross-detector association and deduplication pipeline.
    """
    threshold = _validate_iou_threshold(iou_threshold)

    ppe_detections = []
    other_detections = []

    for detection in detections:
        class_name = str(
            detection.get("class_name", "")
        ).strip().casefold()

        if class_name in PPE_CLASS_NAMES:
            ppe_detections.append(detection)
        else:
            other_detections.append(detection)

    retained_ppe = _merge_same_class_duplicates(
        ppe_detections,
        iou_threshold=threshold,
        same_detector_only=True,
    )

    return other_detections + retained_ppe


def _deduplicate_detections(
    raw_detections,
    raw_scene_graph,
    iou_threshold=SAME_CLASS_DUPLICATE_IOU_THRESHOLD,
):
    """
    Retain one highest-confidence prediction per duplicate-object group.

    Cross-class duplicates require a valid scene-graph association.
    Same-class duplicates require sufficient bounding-box overlap.
    Distinct objects remain regardless of their confidence.
    """
    threshold = _validate_iou_threshold(iou_threshold)

    cross_detector_detections = _merge_cross_detector_duplicates(
        raw_detections,
        raw_scene_graph,
    )

    final_detections = _merge_same_class_duplicates(
        cross_detector_detections,
        iou_threshold=threshold,
    )

    return sorted(
        final_detections,
        key=lambda detection: detection["detection_id"],
    )


def _draw_detections(draw, detections):
    """Draw only the final retained detections."""
    detector_colors = {
        "PPE": (31, 107, 84),
        "Toolset": (31, 99, 164),
        "Vehicles": (192, 110, 31),
        "Toolset-35": (128, 76, 160),
    }

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box_xyxy"])

        detector = detection["detector"]
        class_name = detection["class_name"]
        confidence = _confidence(detection)

        color = detector_colors.get(detector, (100, 100, 100))
        label = f"{detector}: {class_name} {confidence:.2f}"

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=3,
        )

        draw.text(
            (x1, max(0, y1 - 14)),
            label,
            fill="white",
            stroke_width=2,
            stroke_fill=color,
        )


def _classify_screening_outcome(findings, model_names):
    """Classify screening results without implying site safety."""
    findings = findings or []
    selected_models = set(model_names)
    finding_count = len(findings)

    if finding_count:
        outcome = "POTENTIAL"
        explanation = (
            f"The screening rules generated {finding_count} potential "
            "finding(s). These are image-based observations and do not "
            "confirm actual hazards or safety violations."
        )
    else:
        coverage_complete = (
            "ppe" in selected_models
            and "vehicles" in selected_models
        )

        if coverage_complete:
            outcome = "NO_SUPPORTED"
            explanation = (
                "No supported finding was generated by the configured "
                "screening rules. This does not establish that the image "
                "or construction site is hazard-free."
            )
        else:
            outcome = "INSUFFICIENT"
            explanation = (
                "No supported finding was generated, and the selected "
                "detectors do not cover all current screening rules."
            )

    return {
        "outcome": outcome,
        "explanation": explanation,
        "finding_count": finding_count,
        "detector_coverage": {
            "missing_ppe_screening": "ppe" in selected_models,
            "worker_machinery_proximity_screening": (
                "ppe" in selected_models
                and "vehicles" in selected_models
            ),
        },
        "high_confidence_alert_available": False,
        "high_confidence_alert_limitation": (
            "No independently validated hazard-specific criterion "
            "currently supports a high-confidence alert."
        ),
        "interpretation_limit": (
            "NO_SUPPORTED does not mean safe. INSUFFICIENT means that "
            "the selected detector configuration does not cover all "
            "current screening rules."
        ),
    }


def _validate_confidence_threshold(value, description):
    """Validate a confidence threshold."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{description} must be between 0 and 1.")


def run_inference(
    image_path,
    confidence=0.25,
    image_size=640,
    model_names=("ppe", "toolset35", "vehicles"),
    confidence_by_model=None,
    confidence_by_class=None,
    model_variants=None,
):
    """Run configured local detectors and generate a screening report."""
    _validate_confidence_threshold(confidence, "Confidence")

    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size <= 0
    ):
        raise ValueError("Image size must be a positive integer.")

    if (
        not model_names
        or any(name not in MODEL_SPECS for name in model_names)
    ):
        raise ValueError("Select one or more configured detector models.")

    if len(set(model_names)) != len(model_names):
        raise ValueError("Detector model names must be unique.")

    model_variants = model_variants or {}

    if not isinstance(model_variants, dict):
        raise ValueError("Model variants must be a dictionary.")

    if any(name not in model_names for name in model_variants):
        raise ValueError(
            "Model variants must refer to selected detectors."
        )

    confidence_by_model = confidence_by_model or {}

    if not isinstance(confidence_by_model, dict):
        raise ValueError(
            "Detector confidence overrides must be a dictionary."
        )

    if any(name not in MODEL_SPECS for name in confidence_by_model):
        raise ValueError(
            "Confidence overrides must use configured detector names."
        )

    for value in confidence_by_model.values():
        _validate_confidence_threshold(
            value,
            "Detector confidence threshold",
        )

    confidence_by_class = confidence_by_class or {}

    if not isinstance(confidence_by_class, dict):
        raise ValueError(
            "Class confidence overrides must be a dictionary."
        )

    if any(name not in MODEL_SPECS for name in confidence_by_class):
        raise ValueError(
            "Class confidence overrides must use configured detector names."
        )

    for class_thresholds in confidence_by_class.values():
        if not isinstance(class_thresholds, dict):
            raise ValueError(
                "Class confidence overrides must be dictionaries."
            )

        for value in class_thresholds.values():
            _validate_confidence_threshold(
                value,
                "Class confidence threshold",
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

    raw_detections = []
    checkpoints = {}
    nms_thresholds = {}

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

        nms_iou_threshold = (
            0.50 if model_key == "vehicles" else 0.70
        )
        nms_thresholds[model_key] = nms_iou_threshold

        variant_name = model_variants.get(model_key, "yolo11n")
        available_variants = MODEL_VARIANTS.get(model_key, {})

        if variant_name not in available_variants:
            raise ValueError(
                f"Model variant {variant_name!r} is not available "
                f"for detector {model_key!r}."
            )

        variant_spec = available_variants[variant_name]
        architecture = variant_spec["architecture"]
        checkpoint = PROJECT_ROOT.parent / variant_spec["checkpoint"]

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{spec['label']} {variant_name} checkpoint not found: "
                f"{checkpoint}"
            )

        checkpoints[model_key] = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "architecture": architecture,
            "variant": variant_name,
        }

        detector_class = RTDETR if architecture == "rtdetr" else YOLO
        model = detector_class(str(checkpoint))

        results = model.predict(
            source=str(image_path),
            conf=prediction_confidence,
            imgsz=image_size,
            iou=nms_iou_threshold,
            device="cpu",
            verbose=False,
        )

        if not results:
            raise RuntimeError(
                f"{spec['label']} detector returned no inference result."
            )

        result = results[0]

        if result.boxes is None:
            continue

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

            coordinates = box.xyxy[0].tolist()

            if len(coordinates) != 4:
                continue

            x1, y1, x2, y2 = map(int, coordinates)

            if x2 <= x1 or y2 <= y1:
                continue

            raw_detections.append({
                "detection_id": (
                    f"detection_{len(raw_detections):05d}"
                ),
                "detector": spec["label"],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(detection_confidence, 5),
                "confidence_threshold": effective_threshold,
                "box_xyxy": [x1, y1, x2, y2],
            })

    raw_scene_graph = build_scene_graph(
        raw_detections,
        width,
        height,
    )

    detections = _deduplicate_detections(
        raw_detections,
        raw_scene_graph,
    )

    scene_graph = build_scene_graph(
        detections,
        width,
        height,
    )

    assessment = assess_safety(scene_graph)
    findings = assessment.get("findings", [])

    screening_outcome = _classify_screening_outcome(
        findings,
        model_names,
    )

    with Image.open(image_path) as source:
        annotated_image = source.convert("RGB")

    draw = ImageDraw.Draw(annotated_image)
    _draw_detections(draw, detections)

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
            "same_class_duplicate_iou_threshold": (
                SAME_CLASS_DUPLICATE_IOU_THRESHOLD
            ),
            "same_class_ppe_duplicate_iou_threshold": (
                SAME_CLASS_DUPLICATE_IOU_THRESHOLD
            ),
            "image_size": image_size,
            "device": "cpu",
            "detectors": list(model_names),
            "model_variants": {
                name: model_variants.get(name, "yolo11n")
                for name in model_names
            },
        },
        "checkpoint_provenance": checkpoints,
        "raw_detections": raw_detections,
        "raw_scene_graph": raw_scene_graph,
        "detections": detections,
        "scene_graph": scene_graph,
        "deduplication_associations": (
            raw_scene_graph.get("object_associations", [])
        ),
        "safety_assessment": assessment,
        "screening_outcome": screening_outcome,
        "detection_summary": {
            "input_detection_count": len(raw_detections),
            "final_detection_count": len(detections),
            "removed_duplicate_count": (
                len(raw_detections) - len(detections)
            ),
        },
        "interpretation_limit": (
            "The report contains retained detections after same-class "
            "and cross-class duplicate suppression. The highest-confidence "
            "prediction is retained for each identified possible duplicate "
            "group. Distinct objects remain regardless of confidence. "
            "Bounding-box overlap and image-based associations do not "
            "prove physical identity. Detector predictions do not establish "
            "actual PPE use, physical distance, machine operation, or "
            "site safety."
        ),
    }

    return annotated_image, report


def save_run(report, output_root=None):
    """Save a structured inference report in a unique run directory."""
    output_root = Path(
        output_root or PROJECT_ROOT / "reports" / "inference_runs"
    )

    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    run_folder = output_root / timestamp
    run_folder.mkdir(parents=False, exist_ok=False)

    report_path = run_folder / "detections.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return run_folder, report_path
