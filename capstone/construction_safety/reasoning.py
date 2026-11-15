
from construction_safety.association import (
    extract_object_associations,
    extract_relationships,
)

from construction_safety.config import (
    MACHINERY_CLASSES,
    MACHINERY_REVIEW_DISTANCE_RATIO,
    PPE_PERSON_CONTAINMENT_THRESHOLD,
)


TOOLSET35_PPE_CLASSES = {
    "gloves",
    "safety glass",
    "safety headphones",
    "safety helmet",
    "safety mask",
    "safety vest",
}

PPE_CLASSES = {
    "boots",
    "gloves",
    "goggles",
    "helmet",
    "vest",
}


def _normalize_class(value):
    return value.strip().lower().replace("_", " ")


def _category(class_name, detector=None):
    name = _normalize_class(class_name)

    if name == "person":
        return "person"

    if name.startswith("no "):
        return "missing_ppe_prediction"

    # Toolset-35 includes Excavator, but the dedicated Vehicles
    # detector is responsible for machinery classification.
    if detector == "Toolset-35" and name == "excavator":
        return "other"

    if name in MACHINERY_CLASSES:
        return "machinery"

    if name in PPE_CLASSES or (
        detector == "Toolset-35" and name in TOOLSET35_PPE_CLASSES
    ):
        return "ppe_prediction"

    return "other"


def build_scene_graph(
    detections,
    image_width,
    image_height,
    machinery_distance_ratio=MACHINERY_REVIEW_DISTANCE_RATIO,
    containment_threshold=PPE_PERSON_CONTAINMENT_THRESHOLD,
):
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if not 0 <= machinery_distance_ratio <= 1:
        raise ValueError("Machinery distance ratio must be between 0 and 1.")

    if not 0 <= containment_threshold <= 1:
        raise ValueError("Containment threshold must be between 0 and 1.")

    nodes = []

    for index, detection in enumerate(detections):
        box = detection.get("bbox_xyxy")

        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(not isinstance(value, (int, float)) for value in box)
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise ValueError(f"Detection {index} has an invalid bounding box.")

        name = detection.get("class_name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Detection {index} has no class name.")

        detector = detection.get("detector")

        nodes.append({
            "object_id": f"object_{index:04d}",
            "detection_index": index,
            "detector": detector,
            "class_id": detection.get("class_id"),
            "class_name": name,
            "object_category": _category(name, detector),
            "confidence": detection.get("confidence"),
            "bbox_xyxy": list(box),
        })

    # Preserve the existing PPE-containment and machinery-proximity
    # relationship logic.
    relationships = extract_relationships(
        nodes,
        image_width,
        image_height,
        machinery_distance_ratio=machinery_distance_ratio,
        containment_threshold=containment_threshold,
    )

    # Associate matching detections from different detectors as
    # possible duplicate physical objects. Original nodes remain intact.
    object_associations = extract_object_associations(nodes)

    return {
        "schema_version": "1.0",
        "image_dimensions": {
            "width": image_width,
            "height": image_height,
        },
        "parameters": {
            "machinery_distance_ratio": machinery_distance_ratio,
            "ppe_containment_threshold": containment_threshold,
            "object_association_iou_threshold": 0.50,
            "status": "Exploratory; not calibrated risk or distance.",
        },
        "nodes": nodes,
        "relationships": relationships,
        "object_associations": object_associations,
    }


def assess_safety(scene_graph):
    """Compose screening, explanations, and recommendations.

    Preserves the legacy safety-assessment report structure.
    """
    from construction_safety.explanations import generate_explanations
    from construction_safety.hazard_screening import screen_hazards
    from construction_safety.recommendations import generate_recommendations

    assessment = screen_hazards(scene_graph)
    findings = assessment["findings"]

    explanations = generate_explanations(findings)
    recommendations = generate_recommendations(findings)

    for finding in findings:
        finding_id = finding["finding_id"]
        finding.update(explanations[finding_id])
        finding.update(recommendations[finding_id])

    return assessment
