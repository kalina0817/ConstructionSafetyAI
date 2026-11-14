"""Evidence-based explanations for construction safety findings."""


def generate_explanations(findings):
    """Return explanation fields keyed by finding ID."""
    results = {}

    for finding in findings:
        rule_id = finding["rule_id"]

        if rule_id == "MISSING_PPE_PREDICTION_REQUIRES_REVIEW":
            explanation = {
                "observation": (
                    f"The detector predicted `{finding['evidence']['class_name']}`."
                ),
                "uncertainty_reasons": [
                    "A detector prediction is not a confirmed visual assessment.",
                    "Occlusion, image quality, and task requirements are unknown.",
                ],
                "limitation": (
                    "A prediction does not establish PPE absence or a violation."
                ),
            }
        elif rule_id == "IMAGE_SPACE_PROXIMITY_REQUIRES_VERIFICATION":
            explanation = {
                "observation": (
                    "Person and machinery detector boxes are close in the image."
                ),
                "uncertainty_reasons": [
                    "Image perspective affects apparent distance.",
                    "Machine operating status and site controls are unknown.",
                    "Detector outputs may be false positives.",
                ],
                "limitation": (
                    "The image cannot establish physical distance, operation, "
                    "a violation, or an unsafe interaction."
                ),
            }
        else:
            raise ValueError(f"Unsupported finding rule: {rule_id}")

        results[finding["finding_id"]] = explanation

    return results
