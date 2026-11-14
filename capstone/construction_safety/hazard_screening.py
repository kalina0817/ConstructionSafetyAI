def _build_hazard_findings(scene_graph):
    """Generate potential findings with automatic review dispositions."""
    nodes = {
        item["object_id"]: item for item in scene_graph["nodes"]
    }
    findings = []

    for node in scene_graph["nodes"]:
        if node["object_category"] != "missing_ppe_prediction":
            continue

        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": "potential_missing_ppe",
            "rule_id": "MISSING_PPE_PREDICTION_REQUIRES_REVIEW",
            "status": "requires_human_review",
            "review_disposition": "review_required",
            "review_reason": (
                "A missing-PPE prediction requires human verification. "
                "The detection alone does not establish that PPE is absent."
            ),
            "source_object_ids": [node["object_id"]],
            "evidence": {
                "detector": node["detector"],
                "class_name": node["class_name"],
                "confidence": node["confidence"],
                "bbox_xyxy": node["bbox_xyxy"],
            },
        })

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
