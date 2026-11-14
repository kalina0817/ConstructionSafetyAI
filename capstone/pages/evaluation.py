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


st.title("Independent scenario evaluation")
st.write(
    "Compare screening outputs with independently annotated image/category "
    "pairs. This page does not create labels or infer a gold answer from model "
    "outputs."
)
st.warning(
    "Keep this scenario test set separate from detector training/validation, "
    "locked detector test data, and threshold tuning. The 40-image person "
    "duplicate audit is not this evaluation set.",
    icon=":material/lock:",
)
st.markdown(
    "Upload the completed annotation CSV and select the directory containing "
    "exactly one current `detections.json` report per annotated image."
)
st.code(
    "image_id,finding_category,ground_truth,rater_1,rater_2\n"
    "scenario_001.jpg,potential_missing_ppe,uncertain,uncertain,0",
    language="csv",
)
st.download_button(
    "Download annotation template",
    data=(
        "image_id,finding_category,ground_truth,rater_1,rater_2\n"
        "scenario_001.jpg,potential_missing_ppe,uncertain,uncertain,0\n"
        "scenario_001.jpg,potential_worker_machinery_interaction,0,0,0\n"
    ),
    file_name="scenario_annotation_template.csv",
    mime="text/csv",
)

annotations_file = st.file_uploader(
    "Upload independent scenario labels",
    type=["csv"],
)
reports_dir = st.text_input(
    "Inference report directory",
    value=str(INFERENCE_ROOT),
)

if st.button(
    "Evaluate annotations",
    type="primary",
    disabled=annotations_file is None,
):
    selected_reports = Path(reports_dir).expanduser()
    if not selected_reports.is_absolute():
        selected_reports = (Path.cwd() / selected_reports).resolve()
    if not selected_reports.is_dir():
        st.error(f"Report directory not found: {selected_reports}")
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

    st.header("Results")
    metrics = report["overall"]
    with st.container(horizontal=True):
        st.metric("Evaluated pairs", metrics["support"])
        st.metric("Precision", _format(metrics["precision"]))
        st.metric("Recall", _format(metrics["recall"]))
        st.metric("F1", _format(metrics["f1"]))
    st.subheader("Per-category metrics")
    st.dataframe(
        [
            {"Category": category, **values}
            for category, values in report["categories"].items()
        ],
        hide_index=True,
        width="stretch",
        alt="Scenario screening metrics by category",
    )
    st.subheader("Annotation coverage and uncertainty exclusions")
    st.json(report["annotation_summary"])
    if "inter_rater_agreement" in report:
        st.subheader("Inter-rater agreement")
        st.json(report["inter_rater_agreement"])
    st.download_button(
        "Download evaluation report",
        data=json.dumps(report, indent=2),
        file_name="integrated_safety_evaluation.json",
        mime="application/json",
    )


def _format(value):
    return "Undefined" if value is None else f"{value:.3f}"
