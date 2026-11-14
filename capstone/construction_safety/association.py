from math import hypot

from construction_safety.config import (
    MACHINERY_REVIEW_DISTANCE_RATIO,
    PPE_PERSON_CONTAINMENT_THRESHOLD,
)


def _intersection_area(first, second):
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _edge_distance(first, second):
    dx = max(0, first[0] - second[2], second[0] - first[2])
    dy = max(0, first[1] - second[3], second[1] - first[3])
    return hypot(dx, dy)


def extract_relationships(
    nodes,
    image_width,
    image_height,
    machinery_distance_ratio=MACHINERY_REVIEW_DISTANCE_RATIO,
    containment_threshold=PPE_PERSON_CONTAINMENT_THRESHOLD,
):
    """
    Extract geometric relationships from normalized scene objects.

    These are image-space relationships, not proof of physical proximity,
    PPE use, machine operation, or an unsafe condition.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if not 0 <= machinery_distance_ratio <= 1:
        raise ValueError("Machinery distance ratio must be between 0 and 1.")

    if not 0 <= containment_threshold <= 1:
        raise ValueError("Containment threshold must be between 0 and 1.")

    if not isinstance(nodes, list):
        raise ValueError("Nodes must be provided as a list.")

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"Node {index} must be a dictionary.")

        box = node.get("bbox_xyxy")
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(not isinstance(value, (int, float)) for value in box)
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise ValueError(f"Node {index} has an invalid bounding box.")

        if "object_category" not in node or "object_id" not in node:
            raise ValueError(
                f"Node {index} must have object_id and object_category."
            )

    diagonal = hypot(image_width, image_height)
    relationships = []

    for person in nodes:
        if person["object_category"] != "person":
            continue

        for item in nodes:
            if item["object_category"] in {
                "ppe_prediction",
                "missing_ppe_prediction",
            }:
                area = _area(item["bbox_xyxy"])
                containment = (
                    _intersection_area(
                        item["bbox_xyxy"],
                        person["bbox_xyxy"],
                    ) / area
                    if area else 0
                )

                if containment >= containment_threshold:
                    relationships.append({
                        "relationship_id": (
                            f"relationship_{len(relationships):04d}"
                        ),
                        "source_id": item["object_id"],
                        "predicate": "inside_person_bbox",
                        "target_id": person["object_id"],
                        "evidence": {
                            "containment_ratio": round(containment, 5),
                            "threshold": containment_threshold,
                        },
                        "interpretation_limit": (
                            "Bounding-box containment does not establish that "
                            "PPE is worn by this person."
                        ),
                    })

            if item["object_category"] != "machinery":
                continue

            distance = _edge_distance(
                person["bbox_xyxy"],
                item["bbox_xyxy"],
            )
            normalized = distance / diagonal

            if normalized > machinery_distance_ratio:
                continue

            relationships.append({
                "relationship_id": (
                    f"relationship_{len(relationships):04d}"
                ),
                "source_id": person["object_id"],
                "predicate": "near_machinery_in_image",
                "target_id": item["object_id"],
                "evidence": {
                    "edge_distance_pixels": round(distance, 2),
                    "normalized_edge_distance": round(normalized, 5),
                    "threshold": machinery_distance_ratio,
                    "boxes_overlap": (
                        _intersection_area(
                            person["bbox_xyxy"],
                            item["bbox_xyxy"],
                        ) > 0
                    ),
                },
                "interpretation_limit": (
                    "Image-space geometry does not establish physical "
                    "distance, machine operation, an exclusion zone, or "
                    "a hazard."
                ),
            })

    return relationships
