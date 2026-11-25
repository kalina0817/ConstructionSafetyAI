import json
import tempfile
from pathlib import Path

import streamlit as st

from evaluate_safety_system import (
    evaluate,
    load_gold_annotations,
    load_system_predictions,
)
from evaluation_metrics import cohen_kappa


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = ROOT / "reports" / "scenario_evaluation_runs"


def _format_metric(value):
    return "Not available" if value is None else f"{value:.3f}"


st.title("Research and testing")

st.write("How well does the app spot a possible hazard, and when should it say it cannot tell?")
st.subheader("How to make a fair comparison")
st.markdown(
    "1. Choose photos the app has never trained on.\n"
    "2. Ask reviewers to mark each concern as visible, absent, or unclear before seeing the app’s answers.\n"
    "3. Give each method the same photos and keep its settings fixed.\n"
    "4. Count what each method found, what it missed, and where the photo was unclear."
)
with st.expander("How image labels are recorded", expanded=False):
    st.write(
        """
        Each row describes one image and one type of possible safety concern.

        - **1:** The reviewer says the concern is present.
        - **0:** The reviewer says the concern is absent.
        - **uncertain:** The image does not provide enough evidence to decide.

        The check covers missing protective gear and possible
        worker-and-machinery interactions.

        A missing AI detection does not automatically mean that a concern
        is absent. Human labels are used as the comparison reference.
        """
    )

    st.code(
        "image_id,finding_category,ground_truth,rater_1,rater_2\n"
        "scenario_001.jpg,potential_missing_ppe,uncertain,uncertain,0",
        language="csv",
    )

    template = (
        "image_id,finding_category,ground_truth,rater_1,rater_2\n"
        "scenario_001.jpg,potential_missing_ppe,uncertain,uncertain,0\n"
        "scenario_001.jpg,potential_worker_machinery_interaction,0,0,0\n"
    )

    st.download_button(
        "Download image-label template",
        data=template,
        file_name="scenario_annotation_template.csv",
        mime="text/csv",
    )

st.header("Compare the app with a reviewer")

st.warning(
    """
    Use photos reviewed separately for safety. Leave out photos used to test object finding and the 40 photos used to check for repeated labels.
    """,
    icon=":material/lock:",
)

annotations_file = st.file_uploader(
    "Upload the completed image-label CSV",
    type=["csv"],
)

reports_dir_text = st.text_input(
    "Folder containing the detection reports",
    value=str(DEFAULT_REPORTS_DIR),
    help=(
        "The folder must contain one current detections.json report "
        "for each scenario. Older reports may not contain the required "
        "safety assessment."
    ),
)

if st.button(
    "Compare answers",
    type="primary",
    disabled=annotations_file is None,
):
    reports_dir = Path(reports_dir_text).expanduser()

    if not reports_dir.is_absolute():
        reports_dir = (ROOT / reports_dir).resolve()

    if not reports_dir.is_dir():
        st.error(f"Detection report folder not found: {reports_dir}")
        st.stop()

    try:
        with tempfile.TemporaryDirectory(
            prefix="construction-safety-annotations-"
        ) as temp_dir:
            annotation_path = Path(temp_dir) / "annotations.csv"
            annotation_path.write_bytes(annotations_file.getvalue())
            annotations, raters = load_gold_annotations(annotation_path)

        predictions = load_system_predictions(
            reports_dir,
            {image_id for image_id, _ in annotations},
        )

        report = evaluate(annotations, predictions)

        if "rater_1" in raters and "rater_2" in raters:
            report["inter_rater_agreement"] = {
                "measure": "Cohen's kappa",
                "paired_labels": len(raters["rater_1"]),
                "excluded_uncertain_pairs": raters[
                    "excluded_uncertain_pairs"
                ],
                "value": (
                    cohen_kappa(raters["rater_1"], raters["rater_2"])
                    if raters["rater_1"] else None
                ),
                "scope": "Paired binary image-level annotations.",
            }

    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Evaluation failed: {error}")
        st.stop()

    st.subheader("Results")

    metrics = report["overall"]

    with st.container(horizontal=True):
        st.metric("Checks scored", metrics["support"])
        st.metric("Share of alerts reviewers confirmed", _format_metric(metrics["precision"]))
        st.metric("Share of known concerns found", _format_metric(metrics["recall"]))
        st.metric("Combined score", _format_metric(metrics["f1"]))

    with st.expander("What do these numbers mean?"):
        st.markdown(
            """
            - **Alerts reviewers confirmed:** Of the concerns the AI reported, how many
              were also labeled as present by reviewers?
            - **Known concerns found:** Of the concerns reviewers labeled as present,
              how many did the AI find?
            - **Combined score:** This gives equal weight to finding concerns and raising accurate alerts. A higher score means a better balance.

            Scores range from 0 to 1. Higher is generally better.
            """
        )

    st.write("**Results for each safety category**")

    st.dataframe(
        [
            {"Category": category, **values}
            for category, values in report["categories"].items()
        ],
        hide_index=True,
        width="stretch",
        alt="AI results for each supported safety category",
    )

    st.write("**How many labels were used or excluded?**")
    st.json(report["annotation_summary"])

    if "inter_rater_agreement" in report:
        st.write("**How closely did the reviewers agree?**")
        st.write(
            "Cohen's kappa measures how closely two reviewers agreed, "
            "after accounting for agreement that could happen by chance. "
            "A value closer to 1 means stronger agreement. A value near "
            "0 means little agreement beyond chance."
        )
        st.json(report["inter_rater_agreement"])

    st.download_button(
        "Download evaluation results",
        data=json.dumps(report, indent=2),
        file_name="integrated_safety_evaluation.json",
        mime="application/json",
    )

st.caption(
    "Evaluation results depend on the quality and number of human labels. "
    "They measure performance on the reviewed images and do not prove "
    "that the system can detect every real-world safety concern."
)
