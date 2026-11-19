
import math


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

PPE_CATEGORIES = {
    "missing_ppe_prediction",
}

VALID_PROXIMITY_TYPES = {
    "machinery_image_space",
    "tool_image_space",
}


def _get_box(node):
    """Return the canonical bounding-box field."""
    return node.get("box_xyxy")


def _normalized_class_name(node):
    """Normalize class names consistently."""
    name = node.get("class_name", "")

    if not isinstance(name, str):
        return ""

    return " ".join(
        name.strip().casefold().replace("_", " ").split()
    )


def _is_tool(node):
    """Identify supported Toolset tool detections."""
    return (
        node.get("object_category") == "other"
        and _normalized_class_name(node) in TOOL_CLASS_NAMES
    )


def _is_finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _valid_box(box):
    """Check whether a canonical XYXY bounding box is valid."""
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(_is_finite_number(value) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _valid_relationship_evidence(relationship):
    """Validate evidence required for a relationship predicate."""
    evidence = relationship.get("evidence")

    if not isinstance(evidence, dict):
        return False

    predicate = relationship.get("predicate")

    if predicate == "inside_person_box":
        ratio = evidence.get("containment_ratio")
        threshold = evidence.get("threshold")

        return (
            _is_finite_number(ratio)
            and 0 <= ratio <= 1
            and _is_finite_number(threshold)
            and 0 <= threshold <= 1
            and ratio >= threshold
            and _valid_box(evidence.get("item_box_xyxy"))
            and _valid_box(evidence.get("person_box_xyxy"))
        )

    if predicate == "inside_machine_box":
        ratio = evidence.get("person_containment_ratio")
        tolerance = evidence.get("containment_tolerance")

        return (
            _is_finite_number(ratio)
            and 0 <= ratio <= 1
            and _is_finite_number(tolerance)
            and 0 <= tolerance <= 1
            and ratio >= tolerance
            and _valid_box(evidence.get("person_box_xyxy"))
            and _valid_box(evidence.get("machine_box_xyxy"))
        )

    if predicate == "near":
        distance = evidence.get("normalized_edge_distance")
        threshold = evidence.get("threshold")
        threshold_type = evidence.get("threshold_type")

        if (
            not _is_finite_number(distance)
            or distance < 0
            or not _is_finite_number(threshold)
            or not 0 <= threshold <= 1
            or distance > threshold
            or threshold_type not in VALID_PROXIMITY_TYPES
        ):
            return False

        return (
            _valid_box(evidence.get("person_box_xyxy"))
            and _valid_box(evidence.get("object_box_xyxy"))
        )

    return False


def _valid_relationship_identity(relationship):
    """Check that relationship IDs and endpoints are usable."""
    relationship_id = relationship.get("relationship_id")
    source_id = relationship.get("source_id")
    target_id = relationship.get("target_id")

    return (
        isinstance(relationship_id, str)
        and bool(relationship_id.strip())
        and isinstance(source_id, str)
        and bool(source_id.strip())
        and isinstance(target_id, str)
        and bool(target_id.strip())
        and source_id != target_id
    )


def _valid_confidence(node):
    """Validate a prediction against its recorded inference threshold."""
    confidence = node.get("confidence")
    threshold = node.get("confidence_threshold")

    if (
        not _is_finite_number(confidence)
        or not 0 <= confidence <= 1
        or not _is_finite_number(threshold)
        or not 0 <= threshold <= 1
    ):
        return False

    return confidence >= threshold


def _build_hazard_findings(scene_graph):
    """Generate potential findings linked to validated scene evidence."""
    if not isinstance(scene_graph, dict):
        raise ValueError("Scene graph must be a dictionary.")

    raw_nodes = scene_graph.get("nodes")
    raw_relationships = scene_graph.get("relationships")

    if not isinstance(raw_nodes, list):
        raise ValueError("Scene graph nodes must be a list.")

    if not isinstance(raw_relationships, list):
        raise ValueError("Scene graph relationships must be a list.")

    nodes = {}

    for index, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            raise ValueError(
                f"Scene graph node {index} must be a dictionary."
            )

        object_id = item.get("object_id")

        if (
            not isinstance(object_id, str)
            or not object_id.strip()
        ):
            raise ValueError(
                f"Scene graph node {index} has an invalid ID."
            )

        if object_id in nodes:
            raise ValueError(
                f"Duplicate scene graph object ID: {object_id}"
            )

        nodes[object_id] = item

    # Only relationships with valid endpoints, identities, and
    # predicate-specific evidence can contribute to findings.
    relationships = []

    seen_relationship_ids = set()

    for relationship in raw_relationships:
        if not isinstance(relationship, dict):
            continue

        if not _valid_relationship_identity(relationship):
            continue

        relationship_id = relationship["relationship_id"]

        if relationship_id in seen_relationship_ids:
            continue

        if (
            relationship["source_id"] not in nodes
            or relationship["target_id"] not in nodes
        ):
            continue

        if not _valid_relationship_evidence(relationship):
            continue

        seen_relationship_ids.add(relationship_id)
        relationships.append(relationship)

    findings = []

    # Identify people fully contained within machinery boxes.
    # This prevents duplicate machinery proximity findings.
    contained_pairs = {
        (
            relationship["source_id"],
            relationship["target_id"],
        )
        for relationship in relationships
        if (
            relationship["predicate"] == "inside_machine_box"
            and nodes[relationship["source_id"]].get(
                "object_category"
            ) == "person"
            and nodes[relationship["target_id"]].get(
                "object_category"
            ) == "machinery"
        )
    }

    # Index missing-PPE predictions associated with people.
    ppe_person_relationships = {}

    for relationship in relationships:
        if relationship["predicate"] != "inside_person_box":
            continue

        ppe_id = relationship["source_id"]
        person_id = relationship["target_id"]

        ppe_node = nodes[ppe_id]
        person_node = nodes[person_id]

        if (
            ppe_node.get("object_category")
            not in PPE_CATEGORIES
            or person_node.get("object_category") != "person"
        ):
            continue

        ppe_person_relationships.setdefault(
            ppe_id,
            [],
        ).append(relationship)

    # Generate missing-PPE potential findings.
    for node in raw_nodes:
        if (
            node.get("object_category")
            != "missing_ppe_prediction"
        ):
            continue

        if not _valid_confidence(node):
            continue

        candidates = ppe_person_relationships.get(
            node["object_id"],
            [],
        )

        if not candidates:
            continue

        relationship = max(
            candidates,
            key=lambda item: item["evidence"][
                "containment_ratio"
            ],
        )

        person = nodes[relationship["target_id"]]

        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": "potential_missing_ppe",
            "rule_id": (
                "MISSING_PPE_PREDICTION_REQUIRES_VERIFICATION"
            ),
            "status": "potential",
            "review_disposition": "no_review_required",
            "review_reason": (
                "The PPE detector generated a missing-PPE prediction "
                "associated spatially with a person. The system records "
                "it automatically. The prediction does not establish "
                "that PPE is absent."
            ),
            "source_object_ids": [
                node["object_id"],
                person["object_id"],
            ],
            "relationship_id": relationship["relationship_id"],
            "evidence": {
                "detector": node.get("detector"),
                "class_name": node.get("class_name"),
                "confidence": node.get("confidence"),
                "confidence_threshold": node.get(
                    "confidence_threshold"
                ),
                "box_xyxy": _get_box(node),
                "person_box_xyxy": _get_box(person),
                **relationship["evidence"],
            },
            "uncertainty_reasons": [
                "The detector prediction may be incorrect.",
                "Box containment does not confirm PPE absence.",
                (
                    "Image quality, visibility, and occlusion may "
                    "affect detection."
                ),
            ],
            "interpretation_limit": (
                "This is a potential missing-PPE prediction, not a "
                "confirmed PPE violation."
            ),
        })

    # Generate proximity findings.
    # Each person-object pair is emitted at most once per category.
    seen_proximity_pairs = set()

    for relationship in relationships:
        if relationship["predicate"] != "near":
            continue

        person_id = relationship["source_id"]
        object_id = relationship["target_id"]

        person = nodes[person_id]
        related_object = nodes[object_id]

        if person.get("object_category") != "person":
            continue

        category = related_object.get("object_category")
        threshold_type = relationship["evidence"].get(
            "threshold_type"
        )

        if (
            category == "machinery"
            and threshold_type == "machinery_image_space"
        ):
            finding_category = (
                "potential_worker_machinery_proximity"
            )
            rule_id = "PERSON_MACHINERY_IMAGE_PROXIMITY"

        elif (
            _is_tool(related_object)
            and threshold_type == "tool_image_space"
        ):
            finding_category = (
                "potential_worker_tool_proximity"
            )
            rule_id = "PERSON_TOOL_IMAGE_PROXIMITY"

        else:
            continue

        if (
            category == "machinery"
            and (person_id, object_id) in contained_pairs
        ):
            continue

        pair_key = (
            person_id,
            object_id,
            finding_category,
        )

        if pair_key in seen_proximity_pairs:
            continue

        seen_proximity_pairs.add(pair_key)

        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": finding_category,
            "rule_id": rule_id,
            "status": "potential",
            "review_disposition": "no_review_required",
            "review_reason": (
                "A detected person and object satisfy the configured "
                "image-space proximity rule. The system records this "
                "potential spatial finding automatically. Proximity "
                "does not establish contact, tool use, machine operation, "
                "or an unsafe condition."
            ),
            "source_object_ids": [
                person_id,
                object_id,
            ],
            "relationship_id": relationship["relationship_id"],
            "evidence": {
                "person_box_xyxy": _get_box(person),
                "object_box_xyxy": _get_box(related_object),
                "object_class": related_object.get("class_name"),
                "object_category": category,
                **relationship["evidence"],
            },
            "uncertainty_reasons": [
                "Image-space proximity is not physical distance.",
                "The person's activity and intention are unknown.",
                (
                    "The relationship does not establish a "
                    "safety violation."
                ),
            ],
            "interpretation_limit": (
                "This is a potential spatial finding, not proof of "
                "tool use, machine operation, or danger."
            ),
        })

    return findings


def screen_hazards(scene_graph):
    """Apply automated screening rules without claiming confirmed hazards."""
    findings = _build_hazard_findings(scene_graph)

    disposition_counts = {
        "no_review_required": 0,
    }

    for finding in findings:
        disposition = finding["review_disposition"]

        disposition_counts[disposition] = (
            disposition_counts.get(disposition, 0) + 1
        )

    return {
        "schema_version": "1.0",
        "findings": findings,
        "review_triage": {
            "schema_version": "1.0",
            "automatic": True,
            "disposition_counts": disposition_counts,
            "review_queue": [],
            "interpretation_limit": (
                "The system processes images automatically. Human review "
                "is optional and may help verify uncertain findings. "
                "No finding does not establish that a site is safe."
            ),
        },
        "unassessed_conditions": [
            "Fall protection, guardrails, and elevated edges.",
            "Excavation protection, rebar exposure, and environment.",
            "Actual machine operating status and physical separation.",
            "Whether PPE is actually being worn.",
            "Whether a person is holding or using a tool.",
            "Whether a person is working on an object or structure.",
            "Whether a person is operating or controlling machinery.",
            "Standing posture and worker activity.",
        ],
        "risk_assessment": {
            "risk_score": None,
            "risk_probability": None,
            "reason": (
                "Operating conditions, exposure, site controls, and "
                "calibrated outcome labels are unavailable."
            ),
        },
        "interpretation_limit": (
            "These are automated, rule-triggered screening findings, "
            "not confirmed incidents, violations, or unsafe conditions."
        ),
    }
