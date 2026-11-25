
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
    "Check the machinery's operating status and applicable site exclusion "
    "zones. Maintain appropriate separation procedures and confirm that "
    "workers are following the site's machinery safety requirements."
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
        box = detection.get("box_xyxy")

        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(
                not isinstance(value, (int, float))
                for value in box
            )
        ):
            raise ValueError(
                f"Detection {index} has an invalid box_xyxy: {box!r}"
            )

        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(
                f"Detection {index} has a zero-area box."
            )

        class_name = detection.get("class_name")

        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(
                f"Detection {index} has no class_name."
            )

        nodes.append({
            "object_id": f"object_{index:04d}",
            "detection_index": index,
            "source_detection_index": detection.get(
                "source_detection_index", index
            ),
            "detector": detection.get("detector"),
            "class_id": detection.get("class_id"),
            "class_name": class_name,
            "object_category": _object_category(class_name),
            "confidence": detection.get("confidence"),
            "box_xyxy": list(box),
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
                item_area = _box_area(item["box_xyxy"])

                containment = (
                    _intersection_area(
                        item["box_xyxy"],
                        person["box_xyxy"],
                    ) / item_area
                    if item_area > 0
                    else 0.0
                )

                if containment >= ppe_person_containment_threshold:
                    relationships.append({
                        "relationship_id": (
                            f"relationship_{len(relationships):04d}"
                        ),
                        "source_id": item["object_id"],
                        "predicate": "inside_person_bbox",
                        "target_id": person["object_id"],
                        "evidence": {
                            "containment_ratio": round(
                                containment, 5
                            ),
                            "threshold": (
                                ppe_person_containment_threshold
                            ),
                        },
                        "interpretation_limit": (
                            "Bounding-box containment does not establish "
                            "that PPE is worn by this person."
                        ),
                    })

            if item["object_category"] != "machinery":
                continue

            edge_distance_pixels = _edge_distance(
                person["box_xyxy"],
                item["box_xyxy"],
            )

            normalized_edge_distance = (
                edge_distance_pixels / image_diagonal
            )

            if (
                normalized_edge_distance
                > machinery_review_distance_ratio
            ):
                continue

            relationships.append({
                "relationship_id": (
                    f"relationship_{len(relationships):04d}"
                ),
                "source_id": person["object_id"],
                "predicate": "near_machinery_in_image",
                "target_id": item["object_id"],
                "evidence": {
                    "edge_distance_pixels": round(
                        edge_distance_pixels, 2
                    ),
                    "normalized_edge_distance": round(
                        normalized_edge_distance, 5
                    ),
                    "review_threshold": (
                        machinery_review_distance_ratio
                    ),
                    "boxes_overlap": (
                        _intersection_area(
                            person["box_xyxy"],
                            item["box_xyxy"],
                        ) > 0
                    ),
                    "center_distance_pixels": round(
                        _center_distance(
                            person["box_xyxy"],
                            item["box_xyxy"],
                        ),
                        2,
                    ),
                },
                "interpretation_limit": (
                    "This is image-space geometry only; it does not "
                    "establish physical distance, machine operation, "
                    "an exclusion zone, or a hazard."
                ),
            })

    return {
        "schema_version": "1.0",
        "image_dimensions": {
            "width": image_width,
            "height": image_height,
        },
        "geometry_method": (
            "Axis-aligned bounding-box containment, edge distance, "
            "and center distance in image pixels."
        ),
        "parameters": {
            "machinery_review_distance_ratio": (
                machinery_review_distance_ratio
            ),
            "ppe_person_containment_threshold": (
                ppe_person_containment_threshold
            ),
            "threshold_status": (
                "Exploratory image-space thresholds; not calibrated "
                "risk scores or physical-distance estimates."
            ),
        },
        "nodes": nodes,
        "relationships": relationships,
        "interpretation_limit": (
            "The graph describes detector predictions and image-space "
            "relationships. It does not independently confirm object "
            "identity, PPE compliance, or a safety hazard."
        ),
    }


def evaluate_safety_rules(scene_graph, *, worker_ppe=None):
    """
    Generate potential safety alerts automatically.

    Human verification is optional. Predictions and image-space
    relationships are not treated as confirmed safety violations.
    """

    nodes = {
        node["object_id"]: node
        for node in scene_graph.get("nodes", [])
    }

    findings = []

    # Screening groups missing-PPE evidence into one finding per worker and item.
    # Legacy experiment callers still evaluate individual detector predictions.
    worker_items = {
        row["evidence_object_ids"][0]: row for row in (worker_ppe or [])
        if row["status"] == "potentially_missing" and row["evidence_object_ids"]
    }
    # Automatic potential missing-PPE alerts.
    for node in nodes.values():
        class_name = node["class_name"].strip().lower()

        if not class_name.startswith("no_"):
            continue

        item = worker_items.get(node["object_id"])
        if worker_ppe is not None and item is None:
            continue
        rule = RECOMMENDATION_RULES.get(class_name)

        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": "potential_missing_ppe",
            "rule_id": "POTENTIAL_MISSING_PPE_PREDICTION",
            "status": "potential",
            "alert_generated_automatically": True,
            "review_required": False,
            "verification_optional": True,
            "source_object_ids": ([item["worker"], *item["evidence_object_ids"]]
                                  if item else [node["object_id"]]),
            "evidence": {
                "detector": node["detector"],
                "class_name": node["class_name"],
                "confidence": node["confidence"],
                "box_xyxy": node["box_xyxy"],
            },
            "observation": (
                rule["observation"]
                if rule
                else (
                    "A detector predicted a class associated with "
                    "potentially missing PPE."
                )
            ),
            "recommendation": (
                rule["recommendation"]
                if rule
                else (
                    "Check the worker's required protective equipment "
                    "and provide suitable PPE if it is missing."
                )
            ),
            "limitation": (
                rule["limitation"]
                if rule
                else (
                    "A detector prediction does not establish that "
                    "PPE is absent or that a rule has been violated."
                )
            ),
            "uncertainty_reasons": [
                "The finding originates from a detector prediction.",
                "Occlusion and image quality may affect the prediction.",
                "Applicable task and site requirements are not established "
                "by this image.",
            ],
        })

    # Automatic potential worker-machinery alerts.
    for relationship in scene_graph.get("relationships", []):
        if (
            relationship.get("predicate")
            != "near_machinery_in_image"
        ):
            continue

        person = nodes.get(relationship.get("source_id"))
        machinery = nodes.get(relationship.get("target_id"))

        if person is None or machinery is None:
            continue

        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": (
                "potential_worker_machinery_interaction"
            ),
            "rule_id": "POTENTIAL_PERSON_MACHINERY_PROXIMITY",
            "status": "potential",
            "alert_generated_automatically": True,
            "review_required": False,
            "verification_optional": True,
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
                    "box_xyxy": person["box_xyxy"],
                },
                "machinery_detection": {
                    "detector": machinery["detector"],
                    "class_name": machinery["class_name"],
                    "confidence": machinery["confidence"],
                    "box_xyxy": machinery["box_xyxy"],
                },
                **relationship["evidence"],
            },
            "observation": (
                "A person and machinery detection fall within the "
                "configured image-space proximity threshold."
            ),
            "recommendation": MACHINERY_RECOMMENDATION,
            "limitation": MACHINERY_LIMITATION,
            "uncertainty_reasons": [
                "Image perspective and camera position affect apparent "
                "distance.",
                "The machine's operating status and designated exclusion "
                "zone are unknown.",
                "Detector outputs may contain false positives.",
            ],
        })

    category_counts = {}

    for finding in findings:
        category = finding["finding_category"]
        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    return {
        "schema_version": "1.0",
        "findings": findings,
        "summary": {
            "total_potential_alerts": len(findings),
            "potential_missing_ppe": category_counts.get(
                "potential_missing_ppe", 0
            ),
            "potential_worker_machinery_interaction": (
                category_counts.get(
                    "potential_worker_machinery_interaction", 0
                )
            ),
            "requires_human_review": 0,
            "verification_optional": len(findings),
        },
        "unassessed_conditions": UNASSESSED_CONDITIONS,
        "risk_assessment": {
            "risk_score": None,
            "risk_probability": None,
            "reason": (
                "Risk is not scored because operating conditions, "
                "site controls, exposure, and calibrated outcome labels "
                "are unavailable."
            ),
        },
        "interpretation_limit": (
            "Potential alerts are generated automatically from detector "
            "predictions and image-space relationships. They are not "
            "confirmed violations or incidents."
        ),
    }


def generate_review_candidates(
    scene_graph,
    cross_model_comparisons=None,
):
    """
    Generate optional verification candidates.

    Includes missing-PPE predictions, person-machinery proximity,
    and unresolved cross-model class disagreements.

    These candidates do not block automatic alert generation.
    """

    nodes = {
        node["object_id"]: node
        for node in scene_graph.get("nodes", [])
    }

    candidates = []

    # --------------------------------------------------------
    # OPTIONAL PPE VERIFICATION
    # --------------------------------------------------------

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
            "box_xyxy": node["box_xyxy"],
            "status": "potential",
            "review_required": False,
            "verification_optional": True,
            "evidence": (
                "A detector predicted a class associated with "
                "potentially missing PPE."
            ),
            "interpretation": (
                "This prediction does not confirm that PPE is absent."
            ),
        })

    # --------------------------------------------------------
    # OPTIONAL PERSON-MACHINERY VERIFICATION
    # --------------------------------------------------------

    for relationship in scene_graph.get("relationships", []):
        if (
            relationship.get("predicate")
            != "near_machinery_in_image"
        ):
            continue

        person = nodes.get(relationship.get("source_id"))
        machinery = nodes.get(relationship.get("target_id"))

        if person is None or machinery is None:
            continue

        image_dimensions = scene_graph["image_dimensions"]

        image_diagonal = hypot(
            image_dimensions["width"],
            image_dimensions["height"],
        )

        center_distance = relationship["evidence"][
            "center_distance_pixels"
        ]

        candidates.append({
            "candidate_type": (
                "person_machinery_proximity_review"
            ),
            "candidate_index": len(candidates),
            "person_analysis_index": person["detection_index"],
            "person_source_detection_index": (
                person["source_detection_index"]
            ),
            "person_class": person["class_name"],
            "person_detector": person["detector"],
            "person_confidence": person["confidence"],
            "person_box_xyxy": person["box_xyxy"],
            "machinery_analysis_index": (
                machinery["detection_index"]
            ),
            "machinery_source_detection_index": (
                machinery["source_detection_index"]
            ),
            "machinery_class": machinery["class_name"],
            "machinery_detector": machinery["detector"],
            "machinery_confidence": machinery["confidence"],
            "machinery_box_xyxy": machinery["box_xyxy"],
            "center_distance_pixels": center_distance,
            "normalized_image_distance": round(
                center_distance / image_diagonal,
                5,
            ),
            "normalized_edge_distance": relationship["evidence"][
                "normalized_edge_distance"
            ],
            "status": "potential",
            "review_required": False,
            "verification_optional": True,
            "evidence": (
                "Person and machinery detections fall within the "
                "configured image-space proximity threshold."
            ),
            "interpretation": (
                "This relationship does not establish physical distance, "
                "unsafe proximity, machine operation, or a confirmed hazard."
            ),
        })

    # --------------------------------------------------------
    # OPTIONAL CROSS-MODEL CLASS-DISAGREEMENT VERIFICATION
    # --------------------------------------------------------

    for comparison in cross_model_comparisons or []:
        if (
            comparison.get("relationship")
            != "overlap_requires_review"
        ):
            continue

        candidates.append({
            "candidate_type": (
                "cross_model_class_disagreement_review"
            ),
            "candidate_index": len(candidates),

            "analysis_detection_index_a": (
                comparison.get("detection_index_a")
            ),
            "analysis_detection_index_b": (
                comparison.get("detection_index_b")
            ),

            "source_detection_index_a": (
                comparison.get("source_detection_index_a")
            ),
            "source_detection_index_b": (
                comparison.get("source_detection_index_b")
            ),

            "detector_a": comparison.get("detector_a"),
            "class_a": comparison.get("class_a"),
            "confidence_a": comparison.get("confidence_a"),

            "detector_b": comparison.get("detector_b"),
            "class_b": comparison.get("class_b"),
            "confidence_b": comparison.get("confidence_b"),

            "iou": comparison.get("iou"),
            "relationship": comparison.get("relationship"),
            "status": "potential",
            "review_required": False,
            "verification_optional": True,

            "evidence": (
                "Two detectors produced overlapping bounding boxes "
                "but assigned different class labels."
            ),

            "interpretation": (
                "The overlapping predictions may refer to the same "
                "physical object, but overlap alone cannot establish "
                "object identity or determine which class is correct."
            ),
        })

    return {
        "candidate_count": len(candidates),
        "candidates": candidates,
        "review_required": False,
        "verification_optional": True,
        "interpretation_note": (
            "Potential alerts are generated automatically. These records "
            "are available for optional verification and are not required "
            "before alerts can be displayed. Cross-model class disagreements "
            "remain unresolved until verified."
        ),
    }
