from math import hypot

from ppe_safety_rules import RECOMMENDATION_RULES


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

PPE_OBJECT_CLASSES = {
    "boots",
    "gloves",
    "goggles",
    "helmet",
    "vest",
    "safety glass",
    "safety headphones",
    "safety helmet",
    "safety mask",
    "safety vest",
}

TOOLSET_TOOL_CLASSES = {
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
    "wheel barrow",
    "wrench",
}

MACHINERY_REVIEW_DISTANCE_RATIO = 0.05
PPE_PERSON_CONTAINMENT_THRESHOLD = 0.80

MACHINERY_RECOMMENDATION = (
    "Verify whether the equipment is operating, consult the site's "
    "designated exclusion zone, and confirm the worker's actual position "
    "with a qualified site representative."
)
MACHINERY_LIMITATION = (
    "A single image and bounding boxes cannot establish operating status, "
    "real-world distance, an exclusion-zone boundary, or an unsafe interaction."
)
UNASSESSED_CONDITIONS = [
    "Elevated edges, guardrails, and fall-protection use are not assessed.",
    "Excavation protection, rebar exposure, and environmental conditions "
    "are not assessed.",
    "Machine operating status and real-world worker-to-machine distance "
    "are not assessed.",
]


def _normalized_class(class_name):
    return class_name.strip().lower().replace("_", " ")


def _object_category(class_name):
    normalized = _normalized_class(class_name)
    if normalized == "person":
        return "person"
    if normalized.startswith("no "):
        return "missing_ppe_prediction"
    if normalized in MACHINERY_CLASSES:
        return "machinery"
    if normalized in PPE_OBJECT_CLASSES:
        return "ppe_prediction"
    if normalized in TOOLSET_TOOL_CLASSES:
        return "tool_or_equipment"
    return "other"


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _edge_distance(first, second):
    horizontal_gap = max(
        0,
        first[0] - second[2],
        second[0] - first[2],
    )
    vertical_gap = max(
        0,
        first[1] - second[3],
        second[1] - first[3],
    )
    return hypot(horizontal_gap, vertical_gap)


def _center_distance(first, second):
    first_x = (first[0] + first[2]) / 2
    first_y = (first[1] + first[3]) / 2
    second_x = (second[0] + second[2]) / 2
    second_y = (second[1] + second[3]) / 2
    return hypot(first_x - second_x, first_y - second_y)


def build_scene_graph(
    detections,
    image_width,
    image_height,
    machinery_review_distance_ratio=MACHINERY_REVIEW_DISTANCE_RATIO,
    ppe_person_containment_threshold=PPE_PERSON_CONTAINMENT_THRESHOLD,
):
    """Build an evidence-linked graph using image-space box geometry only."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not 0 <= machinery_review_distance_ratio <= 1:
        raise ValueError(
            "Machinery review-distance ratio must be between 0 and 1."
        )
    if not 0 <= ppe_person_containment_threshold <= 1:
        raise ValueError(
            "PPE containment threshold must be between 0 and 1."
        )

    image_diagonal = hypot(image_width, image_height)
    nodes = []
    for index, detection in enumerate(detections):
        box = detection.get("bbox_xyxy")
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(not isinstance(value, (int, float)) for value in box)
        ):
            raise ValueError(
                f"Detection {index} has an invalid bbox_xyxy: {box!r}"
            )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Detection {index} has a zero-area box.")

        class_name = detection.get("class_name")
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(f"Detection {index} has no class_name.")

        nodes.append({
            "object_id": f"object_{index:04d}",
            "detection_index": index,
            "source_detection_index": detection.get("source_detection_index", index),
            "detector": detection.get("detector"),
            "class_id": detection.get("class_id"),
            "class_name": class_name,
            "object_category": _object_category(class_name),
            "confidence": detection.get("confidence"),
            "bbox_xyxy": list(box),
        })

    relationships = []
    for person in nodes:
        if person["object_category"] != "person":
            continue

        for item in nodes:
            if item["object_category"] in {
                "ppe_prediction",
                "missing_ppe_prediction",
            }:
                item_area = _box_area(item["bbox_xyxy"])
                containment = (
                    _intersection_area(
                        item["bbox_xyxy"], person["bbox_xyxy"]
                    ) / item_area
                    if item_area > 0 else 0.0
                )
                if containment >= ppe_person_containment_threshold:
                    relationships.append({
                        "relationship_id": f"relationship_{len(relationships):04d}",
                        "source_id": item["object_id"],
                        "predicate": "inside_person_bbox",
                        "target_id": person["object_id"],
                        "evidence": {
                            "containment_ratio": round(containment, 5),
                            "threshold": ppe_person_containment_threshold,
                        },
                        "interpretation_limit": (
                            "Bounding-box containment does not establish that "
                            "PPE is worn by this person."
                        ),
                    })

            if item["object_category"] != "machinery":
                continue

            edge_distance_pixels = _edge_distance(
                person["bbox_xyxy"], item["bbox_xyxy"]
            )
            normalized_edge_distance = edge_distance_pixels / image_diagonal
            if normalized_edge_distance > machinery_review_distance_ratio:
                continue

            relationships.append({
                "relationship_id": f"relationship_{len(relationships):04d}",
                "source_id": person["object_id"],
                "predicate": "near_machinery_in_image",
                "target_id": item["object_id"],
                "evidence": {
                    "edge_distance_pixels": round(edge_distance_pixels, 2),
                    "normalized_edge_distance": round(
                        normalized_edge_distance, 5
                    ),
                    "review_threshold": machinery_review_distance_ratio,
                    "boxes_overlap": (
                        _intersection_area(
                            person["bbox_xyxy"], item["bbox_xyxy"]
                        ) > 0
                    ),
                    "center_distance_pixels": round(
                        _center_distance(
                            person["bbox_xyxy"], item["bbox_xyxy"]
                        ),
                        2,
                    ),
                },
                "interpretation_limit": (
                    "This is image-space geometry only; it does not establish "
                    "physical distance, machine operation, an exclusion zone, "
                    "or a hazard."
                ),
            })

    return {
        "schema_version": "1.0",
        "image_dimensions": {
            "width": image_width,
            "height": image_height,
        },
        "geometry_method": (
            "Axis-aligned bounding-box containment, edge distance, and "
            "center distance in image pixels."
        ),
        "parameters": {
            "machinery_review_distance_ratio": (
                machinery_review_distance_ratio
            ),
            "ppe_person_containment_threshold": (
                ppe_person_containment_threshold
            ),
            "threshold_status": (
                "Exploratory image-space thresholds; not calibrated risk "
                "scores or physical-distance estimates."
            ),
        },
        "nodes": nodes,
        "relationships": relationships,
        "interpretation_limit": (
            "The graph describes detector predictions and image-space "
            "relationships. It does not independently confirm object identity, "
            "PPE compliance, or a safety hazard."
        ),
    }


def evaluate_safety_rules(scene_graph):
    """Apply explicit screening rules and preserve evidence and uncertainty."""
    nodes = {
        node["object_id"]: node for node in scene_graph.get("nodes", [])
    }
    findings = []

    for node in nodes.values():
        class_name = node["class_name"].strip().lower()
        if not class_name.startswith("no_"):
            continue

        rule = RECOMMENDATION_RULES.get(class_name)
        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": "potential_missing_ppe",
            "rule_id": "PPE_PREDICTION_REQUIRES_VISUAL_REVIEW",
            "status": "requires_human_review",
            "source_object_ids": [node["object_id"]],
            "evidence": {
                "detector": node["detector"],
                "class_name": node["class_name"],
                "confidence": node["confidence"],
                "bbox_xyxy": node["bbox_xyxy"],
            },
            "observation": (
                rule["observation"] if rule else
                "A detector predicted a missing-PPE class."
            ),
            "recommendation": (
                rule["recommendation"] if rule else
                "Have a qualified reviewer verify the visible evidence and "
                "applicable site requirements."
            ),
            "limitation": (
                rule["limitation"] if rule else
                "A detector prediction does not establish that PPE is absent "
                "or that a rule has been violated."
            ),
            "uncertainty_reasons": [
                "The finding originates from a detector prediction.",
                "Occlusion, image quality, and task requirements are not "
                "established by this image.",
            ],
        })

    for relationship in scene_graph.get("relationships", []):
        if relationship["predicate"] != "near_machinery_in_image":
            continue

        person = nodes[relationship["source_id"]]
        machinery = nodes[relationship["target_id"]]
        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": "potential_worker_machinery_interaction",
            "rule_id": "IMAGE_SPACE_PROXIMITY_REQUIRES_VERIFICATION",
            "status": "requires_human_review",
            "source_object_ids": [
                person["object_id"],
                machinery["object_id"],
            ],
            "relationship_id": relationship["relationship_id"],
            "evidence": {
                "person_detection": {
                    "detector": person["detector"],
                    "class_name": person["class_name"],
                    "confidence": person["confidence"],
                    "bbox_xyxy": person["bbox_xyxy"],
                },
                "machinery_detection": {
                    "detector": machinery["detector"],
                    "class_name": machinery["class_name"],
                    "confidence": machinery["confidence"],
                    "bbox_xyxy": machinery["bbox_xyxy"],
                },
                **relationship["evidence"],
            },
            "observation": (
                "A person and machinery detector box are close in the image."
            ),
            "recommendation": MACHINERY_RECOMMENDATION,
            "limitation": MACHINERY_LIMITATION,
            "uncertainty_reasons": [
                "Image perspective and camera position affect apparent "
                "distance.",
                "The machine's operating status and designated exclusion zone "
                "are unknown.",
                "Detector outputs, including machinery detections, may be "
                "false positives.",
            ],
        })

    return {
        "schema_version": "1.0",
        "findings": findings,
        "summary": {
            "potential_missing_ppe": sum(
                item["finding_category"] == "potential_missing_ppe"
                for item in findings
            ),
            "potential_worker_machinery_interaction": sum(
                item["finding_category"]
                == "potential_worker_machinery_interaction"
                for item in findings
            ),
            "requires_human_review": len(findings),
        },
        "unassessed_conditions": UNASSESSED_CONDITIONS,
        "risk_assessment": {
            "risk_score": None,
            "risk_probability": None,
            "reason": (
                "Risk is not scored because operating conditions, site "
                "controls, exposure, and calibrated outcome labels are "
                "unavailable."
            ),
        },
        "interpretation_limit": (
            "These are rule-triggered screening findings, not confirmed "
            "violations or incidents. No finding is a substitute for a "
            "qualified on-site safety assessment."
        ),
    }


def generate_review_candidates(scene_graph):
    """Create review records that retain candidate-to-detection provenance."""
    nodes = {
        node["object_id"]: node for node in scene_graph.get("nodes", [])
    }
    candidates = []

    for node in nodes.values():
        if node["object_category"] != "missing_ppe_prediction":
            continue
        candidates.append({
            "candidate_type": "ppe_prediction_review",
            "candidate_index": len(candidates),
            "analysis_detection_index": node["detection_index"],
            "source_detection_index": node["source_detection_index"],
            "class_name": node["class_name"],
            "confidence": node["confidence"],
            "detector": node["detector"],
            "bbox_xyxy": node["bbox_xyxy"],
            "status": "requires_human_review",
            "evidence": (
                "A detector predicted a class associated with missing PPE."
            ),
            "interpretation": (
                "This detector prediction does not confirm that PPE is absent."
            ),
        })

    for relationship in scene_graph.get("relationships", []):
        if relationship["predicate"] != "near_machinery_in_image":
            continue

        person = nodes[relationship["source_id"]]
        machinery = nodes[relationship["target_id"]]
        candidates.append({
            "candidate_type": "person_machinery_proximity_review",
            "candidate_index": len(candidates),
            "person_analysis_index": person["detection_index"],
            "person_source_detection_index": person["source_detection_index"],
            "person_class": person["class_name"],
            "person_detector": person["detector"],
            "person_confidence": person["confidence"],
            "person_bbox_xyxy": person["bbox_xyxy"],
            "machinery_analysis_index": machinery["detection_index"],
            "machinery_source_detection_index": machinery["source_detection_index"],
            "machinery_class": machinery["class_name"],
            "machinery_detector": machinery["detector"],
            "machinery_confidence": machinery["confidence"],
            "machinery_bbox_xyxy": machinery["bbox_xyxy"],
            "center_distance_pixels": relationship["evidence"][
                "center_distance_pixels"
            ],
            "normalized_image_distance": round(
                relationship["evidence"]["center_distance_pixels"]
                / hypot(
                    scene_graph["image_dimensions"]["width"],
                    scene_graph["image_dimensions"]["height"],
                ),
                5,
            ),
            "normalized_edge_distance": relationship["evidence"][
                "normalized_edge_distance"
            ],
            "status": "requires_human_review",
            "evidence": (
                "Person and machinery detections fall within the exploratory "
                "image-space review threshold."
            ),
            "interpretation": (
                "This relationship does not establish physical distance, "
                "unsafe proximity, machine operation, or a confirmed hazard."
            ),
        })

    return {
        "candidate_count": len(candidates),
        "candidates": candidates,
        "interpretation_note": (
            "Review candidates are generated from detector outputs and "
            "image-space geometry; human review is required."
        ),
    }


