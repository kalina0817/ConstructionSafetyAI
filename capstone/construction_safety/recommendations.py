
"""Direct, action-oriented recommendations for construction safety findings."""

PPE_RECOMMENDATIONS = {
    "no_boots": "Wear appropriate safety boots before continuing work.",
    "no_helmet": "Wear an approved safety helmet before continuing work.",
    "no_gloves": "Wear the required protective gloves before handling equipment or materials.",
    "no_goggles": "Wear appropriate safety goggles before continuing work.",
    "no_goggle": "Wear appropriate safety goggles before continuing work.",
    "no_vest": "Wear a high-visibility safety vest while working on site.",
}

MISSING_PPE_RULES = {
    "MISSING_PPE_PREDICTION_REQUIRES_REVIEW",
    "MISSING_PPE_PREDICTION_REQUIRES_VERIFICATION",
}

PROXIMITY_RULES = {
    "IMAGE_SPACE_PROXIMITY_REQUIRES_VERIFICATION",
    "PERSON_MACHINERY_IMAGE_PROXIMITY",
}


def generate_recommendations(findings):
    """Return direct safety actions keyed by finding ID."""
    results = {}

    for finding in findings:
        rule_id = finding["rule_id"]
        evidence = finding.get("evidence", {})

        if rule_id in MISSING_PPE_RULES:
            class_name = str(evidence.get("class_name", "")).lower()

            recommendation = PPE_RECOMMENDATIONS.get(
                class_name,
                "Wear the required personal protective equipment before continuing work.",
            )

        elif rule_id in PROXIMITY_RULES:
            recommendation = (
                "Verify the person's position relative to the machinery or "
                "vehicle. Maintain a safe separation from operating equipment "
                "and follow the site's designated exclusion-zone procedures."
            )

        else:
            raise ValueError(f"Unsupported finding rule: {rule_id}")

        results[finding["finding_id"]] = {
            "recommendation": recommendation,
        }

    return results
