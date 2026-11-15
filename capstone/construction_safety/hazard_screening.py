
def _build_hazard_findings(scene_graph):
    """Generate evidence-linked potential findings."""
    nodes = {
        item["object_id"]: item
        for item in scene_graph["nodes"]
    }
    findings = []

    # Index qualifying PPE-to-person containment relationships.
    ppe_person_relationships = {}

    for relationship in scene_graph["relationships"]:
        if relationship["predicate"] != "inside_person_bbox":
            continue

        ppe_id = relationship["source_id"]
        person_id = relationship["target_id"]

        ppe_node = nodes.get(ppe_id)
        person_node = nodes.get(person_id)

        if (
            ppe_node is None
            or person_node is None
            or ppe_node["object_category"] != "missing_ppe_prediction"
            or person_node["object_category"] != "person"
        ):
            continue

        ppe_person_relationships.setdefault(ppe_id, []).append(
            relationship
        )

    # Missing-PPE predictions require both threshold-qualified
    # confidence and an association with a detected person.
    for node in scene_graph["nodes"]:
        if node["object_category"] != "missing_ppe_prediction":
            continue

        confidence = node.get("confidence")
        threshold = node.get("confidence_threshold")

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or confidence < threshold
        ):
            continue

        relationships = ppe_person_relationships.get(
            node["object_id"],
            [],
        )

        if not relationships:
            continue

        # Use the strongest containment association to avoid
        # creating duplicate findings for one PPE prediction.
        relationship = max(
            relationships,
            key=lambda item: item["evidence"].get(
                "containment_ratio",
                0,
            ),
        )

        person = nodes[relationship["target_id"]]

        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": "potential_missing_ppe",
            "rule_id": "MISSING_PPE_PREDICTION_REQUIRES_REVIEW",
            "status": "requires_human_review",
            "review_disposition": "review_required",
            "review_reason": (
                "A missing-PPE prediction passed its configured "
                "confidence threshold and was associated with a "
                "detected person. Human verification is still required; "
                "the prediction does not establish that PPE is absent."
            ),
            "source_object_ids": [
                node["object_id"],
                person["object_id"],
            ],
            "relationship_id": relationship["relationship_id"],
            "evidence": {
                "detector": node["detector"],
                "class_name": node["class_name"],
                "confidence": confidence,
                "confidence_threshold": threshold,
                "bbox_xyxy": node["bbox_xyxy"],
                "person_bbox_xyxy": person["bbox_xyxy"],
                **relationship["evidence"],
            },
        })

    # Preserve the existing machinery screening behavior.
    for relationship in scene_graph["relationships"]:
        if relationship["predicate"] != "near_machinery_in_image":
            continue

        person = nodes[relationship["source_id"]]
        machinery = nodes[relationship["target_id"]]

        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": "potential_worker_machinery_interaction",
            "rule_id": "IMAGE_SPACE_PROXIMITY_REQUIRES_VERIFICATION",
            "status": "requires_human_review",
            "review_disposition": "review_required",
            "review_reason": (
                "Image-space proximity requires human verification. "
                "It does not establish physical distance or an unsafe interaction."
            ),
            "source_object_ids": [
                person["object_id"],
                machinery["object_id"],
            ],
            "relationship_id": relationship["relationship_id"],
            "evidence": {
                "person_bbox_xyxy": person["bbox_xyxy"],
                "machinery_bbox_xyxy": machinery["bbox_xyxy"],
                **relationship["evidence"],
            },
        })

    return findings


def screen_hazards(scene_graph):
    """
    Apply rule-based screening and assign automatic review dispositions.

    Findings are screening candidates, not confirmed incidents or violations.
    """
    findings = _build_hazard_findings(scene_graph)

    disposition_counts = {
        "review_required": 0,
        "no_review_required": 0,
        "insufficient_evidence": 0,
    }

    review_queue = []

    for finding in findings:
        disposition = finding["review_disposition"]

        if disposition not in disposition_counts:
            raise ValueError(
                f"Unsupported review disposition: {disposition!r}"
            )

        disposition_counts[disposition] += 1

        if disposition in {
            "review_required",
            "insufficient_evidence",
        }:
            review_queue.append(finding["finding_id"])

    return {
        "schema_version": "1.0",
        "findings": findings,
        "review_triage": {
            "schema_version": "1.0",
            "automatic": True,
            "disposition_counts": disposition_counts,
            "review_queue": review_queue,
            "interpretation_limit": (
                "Findings outside the review queue are not thereby "
                "classified as safe or free of hazards."
            ),
        },
        "unassessed_conditions": [
            "Fall protection, guardrails, and elevated edges.",
            "Excavation protection, rebar exposure, and environment.",
            "Machine operating status and real-world separation.",
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
            "These are rule-triggered screening findings, not confirmed "
            "incidents, violations, or unsafe conditions."
        ),
    }
