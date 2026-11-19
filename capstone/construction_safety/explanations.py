
"""Clear, evidence-based explanations for construction safety findings."""

PPE_NAMES = {
    "boots": "safety boots",
    "helmet": "safety helmet",
    "gloves": "protective gloves",
    "goggles": "safety goggles",
    "goggle": "safety goggles",
    "vest": "high-visibility vest",
}

MISSING_PPE_RULES = {
    "MISSING_PPE_PREDICTION_REQUIRES_REVIEW",
    "MISSING_PPE_PREDICTION_REQUIRES_VERIFICATION",
}

PROXIMITY_RULES = {
    "IMAGE_SPACE_PROXIMITY_REQUIRES_VERIFICATION",
    "PERSON_MACHINERY_IMAGE_PROXIMITY",
}


def generate_explanations(findings):
    """Return user-facing explanation fields keyed by finding ID."""
    results = {}

    for finding in findings:
        rule_id = finding["rule_id"]
        evidence = finding.get("evidence", {})

        if rule_id in MISSING_PPE_RULES:
            class_name = str(evidence.get("class_name", "PPE")).lower()
            normalized_name = class_name.removeprefix("no_")
            equipment = PPE_NAMES.get(
                normalized_name,
                normalized_name.replace("_", " "),
            )

            explanation = {
                "observation": (
                    f"The PPE detector flagged a possible absence of "
                    f"{equipment} for the detected person."
                ),
                "uncertainty_reasons": [
                    "The detector prediction may be incorrect.",
                    "Occlusion, image quality, and visibility may affect detection.",
                    "The person's actual PPE and task requirements are not confirmed.",
                ],
                "limitation": (
                    f"The image cannot confirm whether the detected person is "
                    f"wearing {equipment} or whether a safety violation has occurred."
                ),
            }

        elif rule_id in PROXIMITY_RULES:
            explanation = {
                "observation": (
                    "A person has been detected near machinery or a vehicle "
                    "based on the configured image-space proximity threshold. "
                    "This may indicate a potential struck-by or caught-between "
                    "hazard."
                ),
                "uncertainty_reasons": [
                    "Camera perspective can affect apparent distance.",
                    "Actual physical separation and equipment operating status are unknown.",
                    "Detector predictions may be incorrect.",
                ],
                "limitation": (
                    "The image cannot confirm actual physical distance, "
                    "equipment operation, a safety violation, or an unsafe interaction."
                ),
            }

        else:
            raise ValueError(f"Unsupported finding rule: {rule_id}")

        results[finding["finding_id"]] = explanation

    return results
