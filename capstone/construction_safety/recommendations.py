"""Rule-based prevention recommendations for screening findings."""


def generate_recommendations(findings):
    """Return recommendations keyed by finding ID."""
    results = {}

    for finding in findings:
        rule_id = finding["rule_id"]

        if rule_id == "MISSING_PPE_PREDICTION_REQUIRES_REVIEW":
            recommendation = (
                "Verify whether the equipment is visibly absent, whether the "
                "image permits that judgment, and what the site task requires."
            )
        elif rule_id == "IMAGE_SPACE_PROXIMITY_REQUIRES_VERIFICATION":
            recommendation = (
                "Verify equipment operating status and the site's designated "
                "exclusion zone with a qualified site representative."
            )
        else:
            raise ValueError(f"Unsupported finding rule: {rule_id}")

        results[finding["finding_id"]] = {
            "recommendation": recommendation,
        }

    return results
