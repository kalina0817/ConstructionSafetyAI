
import json
import tempfile
from pathlib import Path

import streamlit as st

from construction_safety.config import INFERENCE_ROOT
from construction_safety.evaluation import (
    cohen_kappa,
    evaluate,
    load_annotations,
    load_predictions,
)


def _format(value):
    return "Undefined" if value is None else f"{value:.3f}"


st.title("Independent scenario evaluation")

st.write(
    "This page compares the app's findings with labels created by human "
    "reviewers. The reviewers should label the images without using the "
    "app's predictions."
)

st.warning(
    "Use a separate group of test images for this evaluation. Do not use "
    "images from model training, model testing, or threshold tuning. "
    "The 40-image check for duplicate person detections is a separate task.",
    icon=":material/lock:",
)

st.markdown(
    "Upload the completed CSV file with the reviewers' labels. "
    "Then select the folder containing the app's `detections.json` "
    "reports for those images."
)

st.code(
    "image_id,finding_category,ground_truth,rater_1,rater_2\n"
    "scenario_001.jpg,potential_missing_ppe,uncertain,uncertain,uncertain",
    language="csv",
)

st.download_button(
    "Download the CSV template",
    data=(
        "image_id,finding_category,ground_truth,rater_1,rater_2\n"
        "scenario_001.jpg,potential_missing_ppe,uncertain,uncertain,uncertain\n"
        "scenario_001.jpg,potential_worker_machinery_interaction,0,0,0\n"
    ),
    file_name="scenario_annotation_template.csv",
    mime="text/csv",
)

annotations_file = st.file_uploader(
    "Upload the reviewers' labels",
    type=["csv"],
)

reports_dir = st.text_input(
    "Folder containing screening reports",
    value=str(INFERENCE_ROOT),
)

if st.button(
    "Run evaluation",
    type="primary",
    disabled=annotations_file is None,
):
    selected_reports = Path(reports_dir).expanduser()

    if not selected_reports.is_absolute():
        selected_reports = (Path.cwd() / selected_reports).resolve()

    if not selected_reports.is_dir():
        st.error(f"Folder not found: {selected_reports}")
        st.stop()

    try:
        with tempfile.TemporaryDirectory(
            prefix="capstone-annotation-upload-"
        ) as temp_dir:
            csv_path = Path(temp_dir) / "scenario_annotations.csv"
            csv_path.write_bytes(annotations_file.getvalue())
            annotations, raters = load_annotations(csv_path)

        predictions = load_predictions(
            selected_reports,
            {image for image, _ in annotations},
        )

        report = evaluate(annotations, predictions)

        if raters:
            report["inter_rater_agreement"] = {
                "measure": "Cohen's kappa",
                "paired_labels": len(raters["rater_1"]),
                "excluded_uncertain_pairs": raters[
                    "excluded_uncertain_pairs"
                ],
                "value": cohen_kappa(
                    raters["rater_1"],
                    raters["rater_2"],
                ),
            }

    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Evaluation failed: {error}")
        st.stop()

    st.header("Evaluation results")

    metrics = report["overall"]

    with st.container(horizontal=True):
        st.metric("Image/category pairs checked", metrics["support"])
        st.metric("Precision", _format(metrics["precision"]))
        st.metric("Recall", _format(metrics["recall"]))
        st.metric("F1 score", _format(metrics["f1"]))

    st.caption(
        "Precision measures how often the app's flagged findings were "
        "correct. Recall measures how many of the labeled findings "
        "the app detected. F1 combines both."
    )

    st.subheader("Results for each category")

    st.dataframe(
        [
            {"Category": category, **values}
            for category, values in report["categories"].items()
        ],
        hide_index=True,
        width="stretch",
        alt="Screening results by category",
    )

    st.subheader("Reviewer labels and uncertain cases")
    st.json(report["annotation_summary"])

    if "inter_rater_agreement" in report:
        st.subheader("Agreement between reviewers")
        st.json(report["inter_rater_agreement"])

    st.download_button(
        "Download evaluation results",
        data=json.dumps(report, indent=2),
        file_name="integrated_safety_evaluation.json",
        mime="application/json",
    )
