RECOMMENDATION_RULES = {
    "no_boots": {
        "observation": "The PPE detector predicts a possible missing protective-footwear condition.",
        "recommendation": (
            "Review the worker's footwear and verify whether task-appropriate "
            "protective footwear is required by site policy."
        ),
        "limitation": (
            "The image alone does not establish the task, footwear "
            "requirements, or whether compliant footwear is obscured."
        ),
    },
    "no_gloves": {
        "observation": "The PPE detector predicts a possible missing safety-gloves condition.",
        "recommendation": (
            "Review the worker's hand protection and determine whether "
            "task-appropriate gloves are required for the activity."
        ),
        "limitation": (
            "The image alone does not establish the task, glove requirements, "
            "or whether alternative hand protection is being used."
        ),
    },
    "no_goggle": {
        "observation": "The PPE detector predicts a possible missing eye-protection condition.",
        "recommendation": (
            "Review the worker's eye protection and verify whether "
            "task-appropriate protection is required by site policy."
        ),
        "limitation": (
            "The image alone does not establish the task, eye-protection "
            "requirements, or whether compliant protection is obscured."
        ),
    },
    "no_helmet": {
        "observation": "The PPE detector predicts a possible missing head-protection condition.",
        "recommendation": (
            "Review the worker's head protection and verify whether "
            "task-appropriate protection is required by site policy."
        ),
        "limitation": (
            "The image alone does not establish the task, head-protection "
            "requirements, or whether compliant protection is obscured."
        ),
    },
    "no_vest": {
        "observation": "The PPE detector predicts a possible missing high-visibility-vest condition.",
        "recommendation": (
            "Review the worker's high-visibility clothing and verify "
            "whether it meets the applicable site requirements."
        ),
        "limitation": (
            "The image alone does not establish the site's required "
            "high-visibility clothing policy or prove a regulatory violation."
        ),
    },
}
