
import json
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
INFERENCE_ROOT = ROOT / "reports" / "inference_runs"


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Construction Safety AI - Human Review",
    layout="wide"
)

st.title("Construction Safety AI")
st.subheader("Human Review of Potential Safety Findings")

st.info(
    "These are model-generated review candidates, not confirmed "
    "PPE violations or safety hazards. Human judgment is required."
)


# --------------------------------------------------
# FIND AVAILABLE INFERENCE RUNS
# --------------------------------------------------

if not INFERENCE_ROOT.exists():
    st.error(f"Inference folder not found: {INFERENCE_ROOT}")
    st.stop()

available_runs = sorted(
    [
        folder
        for folder in INFERENCE_ROOT.iterdir()
        if folder.is_dir()
        and (folder / "detections.json").exists()
    ],
    key=lambda folder: folder.stat().st_mtime,
    reverse=True
)

if not available_runs:
    st.warning("No inference reports are available yet.")
    st.stop()

latest_run = available_runs[0]

st.success(f"Latest inference run: {latest_run.name}")

selected_run = st.selectbox(
    "Select inference run",
    available_runs,
    index=0,
    format_func=lambda folder: folder.name
)

run_folder = selected_run
report_file = run_folder / "detections.json"
annotated_image = run_folder / "unified_detections.jpg"
review_file = run_folder / "human_review.json"


# --------------------------------------------------
# LOAD DETECTION REPORT
# --------------------------------------------------

with report_file.open("r", encoding="utf-8") as file:
    report = json.load(file)

st.write("**Selected report:**")
st.code(str(report_file))

if annotated_image.exists():
    st.image(
        str(annotated_image),
        caption=f"Combined detections - {run_folder.name}",
        width="stretch",
        alt=f"Combined detector predictions for run {run_folder.name}"
    )
else:
    st.warning("Annotated image was not found in this folder.")


# --------------------------------------------------
# LOAD EXISTING HUMAN DECISIONS
# --------------------------------------------------

existing_decisions = {}

if review_file.exists():
    with review_file.open("r", encoding="utf-8") as file:
        saved = json.load(file)

    for item in saved.get("review_decisions", []):
        existing_decisions[item["candidate_index"]] = item

    st.success("Previously saved review decisions loaded.")
else:
    st.info("No human review has been saved for this inference run yet.")


# --------------------------------------------------
# REVIEW CANDIDATES
# --------------------------------------------------

candidates = report.get("review_analysis", {}).get("candidates", [])

st.write(f"**Candidates requiring review: {len(candidates)}**")

if not candidates:
    st.success("There are no review candidates in this report.")

else:
    with st.form("human_review_form"):

        review_decisions = []

        for index, candidate in enumerate(candidates):
            st.divider()
            st.markdown(f"### Candidate {index + 1}")

            st.write("**Type:**", candidate.get("candidate_type"))
            st.write("**Status:**", candidate.get("status"))
            st.write("**Evidence:**", candidate.get("evidence"))
            st.write("**Interpretation:**", candidate.get("interpretation"))

            if candidate.get("class_name"):
                st.write("**Predicted class:**", candidate["class_name"])

            if candidate.get("confidence") is not None:
                st.write("**Confidence:**", candidate["confidence"])

            if candidate.get("person_class"):
                st.write(
                    "**Person:**",
                    candidate["person_class"],
                    f"({candidate['person_confidence']:.3f})"
                )
                st.write(
                    "**Machinery:**",
                    candidate["machinery_class"],
                    f"({candidate['machinery_confidence']:.3f})"
                )
                st.write(
                    "**Normalized image distance:**",
                    candidate["normalized_image_distance"]
                )

            previous = existing_decisions.get(index, {})

            decision_options = [
                "Pending",
                "Confirmed",
                "Rejected",
                "Uncertain"
            ]

            previous_decision = previous.get("decision", "Pending")

            if previous_decision not in decision_options:
                previous_decision = "Pending"

            decision = st.selectbox(
                "Human review decision",
                decision_options,
                index=decision_options.index(previous_decision),
                key=f"decision_{run_folder.name}_{index}"
            )

            note = st.text_area(
                "Reviewer notes",
                value=previous.get("reviewer_notes", ""),
                key=f"note_{run_folder.name}_{index}"
            )

            review_decisions.append({
                "candidate_index": index,
                "candidate": candidate,
                "decision": decision,
                "reviewer_notes": note
            })

        submitted = st.form_submit_button("Save review decisions")

    if submitted:
        review_data = {
            "source_report": str(report_file.resolve()),
            "review_decisions": review_decisions
        }

        with review_file.open("w", encoding="utf-8") as file:
            json.dump(review_data, file, indent=4)

        st.success("Human review decisions saved successfully.")
        st.write("Saved separately to:")
        st.code(str(review_file))

st.divider()

st.caption(
    "Human decisions are stored separately from detector predictions. "
    "Saving a review does not modify the original dataset or model output."
)
