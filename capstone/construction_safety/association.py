
from math import hypot, isfinite

from construction_safety.config import (
    MACHINERY_REVIEW_DISTANCE_RATIO,
    PPE_PERSON_CONTAINMENT_THRESHOLD,
    TOOL_REVIEW_DISTANCE_RATIO,
)


FULL_CONTAINMENT_TOLERANCE = 0.999

PPE_CATEGORIES = {
    "ppe_prediction",
    "missing_ppe_prediction",
}

# Preserve the configured Toolset tool classes.
# Do not classify every object in the Toolset dataset as a tool.
TOOL_CLASS_NAMES = {
    "anvil",
    "aviation snip",
    "axe",
    "brush",
    "caulk gun",
    "chisel",
    "crowbar",
    "drill",
    "flashlight",
    "grinder",
    "hammer",
    "knife",
    "mallet",
    "measuring tape",
    "meter",
    "nails",
    "pickaxe",
    "pliers",
    "rake",
    "ruler",
    "saw",
    "screwdriver",
    "shovel",
    "spade",
    "trowel",
    "utility knife",
    "wrench",
    "wheel barrow",
}


def _get_box(node):
    """Return the canonical bounding-box field."""
    return node.get("box_xyxy")


def _normalized_class_name(node):
    """Normalize class names consistently across detectors."""
    name = node.get("class_name", "")

    if not isinstance(name, str):
        return ""

    return " ".join(
        name.strip().casefold().replace("_", " ").split()
    )


# Shared class equivalences for cross-detector matching.
# Keep these rules in one place so association and inference
# deduplication cannot silently disagree.
CLASS_NAME_ALIASES = {
    "vest": "safety vest",
    "safety vest": "safety vest",
    "helmet": "safety helmet",
    "safety helmet": "safety helmet",
    "gloves": "gloves",
    "goggles": "safety glasses",
    "safety glass": "safety glasses",
    "safety glasses": "safety glasses",
    "shovel": "spade",
    "spade": "spade",
    "knife": "utility knife",
    "utility knife": "utility knife",
    "measuring tape": "measuring tool",
    "meter": "measuring tool",
    "ruler": "measuring tool",
}


def canonical_class_name(class_name):
    """Return a shared class name for cross-detector matching."""
    if not isinstance(class_name, str):
        return None

    name = " ".join(
        class_name.strip().casefold().replace("_", " ").split()
    )

    if (
        not name
        or name == "person"
        or name.startswith("no ")
    ):
        return None

    return CLASS_NAME_ALIASES.get(name, name)


def _is_tool(node):
    """Identify supported Toolset tool detections."""
    return (
        node.get("object_category") == "other"
        and _normalized_class_name(node) in TOOL_CLASS_NAMES
    )


def _intersection_area(first, second):
    width = max(
        0,
        min(first[2], second[2]) - max(first[0], second[0]),
    )
    height = max(
        0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )
    return width * height


def _area(box):
    return max(0, box[2] - box[0]) * max(
        0,
        box[3] - box[1],
    )


def _intersection_over_union(first, second):
    intersection = _intersection_area(first, second)
    union = _area(first) + _area(second) - intersection

    return intersection / union if union > 0 else 0.0


def _containment_ratio(inner, outer):
    inner_area = _area(inner)

    if inner_area <= 0:
        return 0.0

    return _intersection_area(inner, outer) / inner_area


def _edge_distance(first, second):
    """Calculate the minimum distance between two axis-aligned boxes."""
    dx = max(
        0,
        first[0] - second[2],
        second[0] - first[2],
    )
    dy = max(
        0,
        first[1] - second[3],
        second[1] - first[3],
    )

    return hypot(dx, dy)


def _validate_box(box, index):
    """Validate and normalize a box without modifying the source node."""
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in box
        )
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError(
            f"Node {index} has an invalid bounding box."
        )

    return tuple(float(value) for value in box)


def _validate_ratio(value, name):
    """Validate a normalized threshold."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            f"{name} must be a finite number between 0 and 1."
        )

    return float(value)


def _validate_nodes(nodes, require_category=False):
    """Validate node identities and geometry."""
    if not isinstance(nodes, list):
        raise ValueError("Nodes must be provided as a list.")

    seen_ids = set()
    validated_boxes = {}

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(
                f"Node {index} must be a dictionary."
            )

        object_id = node.get("object_id")
        category = node.get("object_category")

        if (
            not isinstance(object_id, str)
            or not object_id.strip()
        ):
            raise ValueError(
                f"Node {index} has an invalid object_id."
            )

        if object_id in seen_ids:
            raise ValueError(
                f"Duplicate object_id: {object_id}"
            )

        seen_ids.add(object_id)

        if require_category and (
            not isinstance(category, str)
            or not category.strip()
        ):
            raise ValueError(
                f"Node {index} has an invalid object_category."
            )

        if category is not None and (
            not isinstance(category, str)
            or not category.strip()
        ):
            raise ValueError(
                f"Node {index} has an invalid object_category."
            )

        validated_boxes[object_id] = _validate_box(
            _get_box(node),
            index,
        )

    return validated_boxes


def extract_object_associations(nodes, iou_threshold=0.50):
    """
    Identify possible duplicate detections across different detectors.

    Associations remain pairwise so downstream union-find deduplication
    can combine transitive matches.
    """
    iou_threshold = _validate_ratio(
        iou_threshold,
        "IoU threshold",
    )

    # Categories are not required for cross-detector object matching.
    validated_boxes = _validate_nodes(nodes)

    associations = []

    for index, first in enumerate(nodes):
        for second_index in range(index + 1, len(nodes)):
            second = nodes[second_index]

            first_detector = first.get("detector")
            second_detector = second.get("detector")

            if (
                not isinstance(first_detector, str)
                or not first_detector.strip()
                or not isinstance(second_detector, str)
                or not second_detector.strip()
                or first_detector == second_detector
            ):
                continue

            # Use the shared canonical class-name rules.
            first_name = canonical_class_name(
                first.get("class_name")
            )
            second_name = canonical_class_name(
                second.get("class_name")
            )

            if (
                first_name is None
                or first_name != second_name
            ):
                continue

            first_box = validated_boxes[first["object_id"]]
            second_box = validated_boxes[second["object_id"]]

            iou = _intersection_over_union(
                first_box,
                second_box,
            )

            if iou < iou_threshold:
                continue

            associations.append({
                "association_id": (
                    f"association_{len(associations):04d}"
                ),
                "member_ids": [
                    first["object_id"],
                    second["object_id"],
                ],
                "predicate": "possible_same_physical_object",
                "evidence": {
                    "intersection_over_union": round(iou, 5),
                    "threshold": iou_threshold,
                    "detectors": [
                        first_detector,
                        second_detector,
                    ],
                    "class_name": first.get("class_name"),
                    "canonical_class_name": first_name,
                },
                "interpretation_limit": (
                    "Box overlap across detectors suggests a possible "
                    "duplicate detection; it does not confirm physical "
                    "object identity."
                ),
            })

    return associations


def extract_relationships(
    nodes,
    image_width,
    image_height,
    machinery_distance_ratio=MACHINERY_REVIEW_DISTANCE_RATIO,
    containment_threshold=PPE_PERSON_CONTAINMENT_THRESHOLD,
    tool_distance_ratio=TOOL_REVIEW_DISTANCE_RATIO,
):
    """
    Extract geometric scene relationships.

    Predicates:
      inside_person_box
      inside_machine_box
      near

    Object associations are generated separately.

    Bounding boxes cannot establish actions such as wearing PPE,
    holding or using tools, or operating machinery.
    """
    for value, name in (
        (image_width, "Image width"),
        (image_height, "Image height"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be positive and finite."
            )

    machinery_distance_ratio = _validate_ratio(
        machinery_distance_ratio,
        "Machinery distance ratio",
    )

    tool_distance_ratio = _validate_ratio(
        tool_distance_ratio,
        "Tool distance ratio",
    )

    containment_threshold = _validate_ratio(
        containment_threshold,
        "Containment threshold",
    )

    # Spatial relationships require a valid category on every node.
    validated_boxes = _validate_nodes(
        nodes,
        require_category=True,
    )

    diagonal = hypot(image_width, image_height)
    relationships = []

    def add_relationship(
        source_id,
        predicate,
        target_id,
        evidence,
        interpretation_limit,
    ):
        relationships.append({
            "relationship_id": (
                f"relationship_{len(relationships):04d}"
            ),
            "source_id": source_id,
            "predicate": predicate,
            "target_id": target_id,
            "evidence": evidence,
            "interpretation_limit": interpretation_limit,
        })

    for person in nodes:
        if person["object_category"] != "person":
            continue

        person_box = validated_boxes[person["object_id"]]

        for item in nodes:
            if item["object_id"] == person["object_id"]:
                continue

            category = item["object_category"]
            item_box = validated_boxes[item["object_id"]]

            # PPE-to-person spatial containment.
            if category in PPE_CATEGORIES:
                containment = _containment_ratio(
                    item_box,
                    person_box,
                )

                if containment >= containment_threshold:
                    add_relationship(
                        item["object_id"],
                        "inside_person_box",
                        person["object_id"],
                        {
                            "containment_ratio": round(
                                containment,
                                5,
                            ),
                            "threshold": containment_threshold,
                            "containment_basis": (
                                "intersection_over_item_area"
                            ),
                            "item_box_xyxy": list(item_box),
                            "person_box_xyxy": list(person_box),
                        },
                        (
                            "Box containment associates this prediction "
                            "spatially with a person. It does not establish "
                            "that PPE is being worn or is absent."
                        ),
                    )

                continue

            # Person fully inside a machinery bounding box.
            if category == "machinery":
                person_containment = _containment_ratio(
                    person_box,
                    item_box,
                )

                if (
                    person_containment
                    >= FULL_CONTAINMENT_TOLERANCE
                ):
                    add_relationship(
                        person["object_id"],
                        "inside_machine_box",
                        item["object_id"],
                        {
                            "person_containment_ratio": round(
                                person_containment,
                                5,
                            ),
                            "containment_tolerance": (
                                FULL_CONTAINMENT_TOLERANCE
                            ),
                            "person_box_xyxy": list(person_box),
                            "machine_box_xyxy": list(item_box),
                        },
                        (
                            "The person's box is contained within the "
                            "machinery box. This does not establish "
                            "machine operation or control."
                        ),
                    )

                    # Do not also emit near for this same pair.
                    continue

            # Only machinery and recognized tools are proximity targets.
            if category == "machinery":
                threshold = machinery_distance_ratio
                threshold_type = "machinery_image_space"
            elif _is_tool(item):
                threshold = tool_distance_ratio
                threshold_type = "tool_image_space"
            else:
                continue

            distance = _edge_distance(
                person_box,
                item_box,
            )
            normalized_distance = distance / diagonal

            if normalized_distance > threshold:
                continue

            add_relationship(
                person["object_id"],
                "near",
                item["object_id"],
                {
                    "edge_distance_pixels": round(
                        distance,
                        2,
                    ),
                    "normalized_edge_distance": round(
                        normalized_distance,
                        5,
                    ),
                    "threshold": threshold,
                    "threshold_type": threshold_type,
                    "boxes_overlap": (
                        _intersection_area(
                            person_box,
                            item_box,
                        ) > 0
                    ),
                    "object_category": category,
                    "object_class": item.get("class_name"),
                    "person_box_xyxy": list(person_box),
                    "object_box_xyxy": list(item_box),
                },
                (
                    "This is image-space proximity to a detected machine "
                    "or tool. It does not establish physical distance, "
                    "tool use, machine operation, or a safety violation."
                ),
            )

    return relationships
