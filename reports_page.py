import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "reports" / "inference_runs"


st.title("Saved photo reports")
st.write(
    "View results from previous construction-site image screenings. "
    "Choose a run to see what the AI detected and what needs further checking."
)


def find_reports():
    reports = []

    if not RUNS_DIR.exists():
        return reports

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        json_path = run_dir / "detections.json"

        full_report = run_dir / "screening_report.json"
        if full_report.is_file():
            json_path = full_report

        if json_path.exists():
            reports.append((run_dir, json_path))

    return sorted(
        reports,
        key=lambda item: item[0].stat().st_mtime,
        reverse=True,
    )


reports = find_reports()

if not reports:
    st.warning("There are no saved screening reports yet.")
    st.caption(f"Reports are expected in: {RUNS_DIR}")
    st.stop()


selected_run = st.selectbox(
    "Choose a saved check",
    reports,
    format_func=lambda item: item[0].name,
)

run_dir, json_path = selected_run

try:
    report_bytes = json_path.read_bytes()
    report = json.loads(report_bytes)
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Could not open this report: {error}")
    st.stop()


outcome = report.get("screening_outcome", {})
safety = report.get("safety_assessment", {})
findings = safety.get("findings", [])
detections = report.get("analysis_detections", [])

outcome_code = outcome.get("outcome", "Unavailable")

outcome_labels = {
    "HIGH_CONFIDENCE_ALERT": "High-confidence alert",
    "POTENTIAL": "Possible concern",
    "INSUFFICIENT": "Not enough information",
    "NO_SUPPORTED": "No supported concern found",
}

outcome_explanations = {
    "HIGH_CONFIDENCE_ALERT": (
        "The app found a strong reason to review this image. "
        "Check the image and site conditions."
    ),
    "POTENTIAL": (
        "The image contains a possible safety concern. "
        "A person should review it before deciding what action is needed."
    ),
    "INSUFFICIENT": (
        "The photo gives too little information "
        "to assess the possible concern."
    ),
    "NO_SUPPORTED": (
        "The app found no concern it could support in this photo. "
        "This does not prove that the site is safe."
    ),
}

display_outcome = outcome_labels.get(outcome_code, outcome_code)

st.header("What the app found")
if report.get("reasoning_model"):
    st.write(f"Found objects with **{report.get('detection_model', 'Unknown')}** · Checked risks with **{report['reasoning_model']}**")
if report.get("reasoning_error"):
    st.error(f"The risk check could not finish. {report['reasoning_error']}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Result", display_outcome)
col2.metric("Possible findings", len(findings))
col3.metric(
    "Objects kept",
    report.get("retained_detection_count", len(detections)),
)
col4.metric(
    "Repeated labels removed",
    report.get("suppressed_duplicate_count", 0),
)

saved_explanation = outcome.get("explanation")
st.write(
    outcome_explanations.get(
        outcome_code,
        saved_explanation or "No explanation was saved for this result.",
    )
)

if saved_explanation and outcome_code not in outcome_explanations:
    st.caption(saved_explanation)

st.info(
    "This is an image-based screening result, not a final safety decision. "
    "Image positions do not prove real-world distance or equipment activity. "
    "A missing PPE detection does not prove that the worker was not wearing PPE."
)

st.divider()

st.header("What the AI found in the image")

image_path = run_dir / "unified_detections.jpg"

if image_path.exists():
    st.image(
        str(image_path),
        caption=f"Image analyzed in run: {run_dir.name}",
        width="stretch",
    )
else:
    st.info("An annotated image was not saved for this run.")

st.divider()

st.header("Possible safety concerns")

if findings:
    for index, finding in enumerate(findings, start=1):
        category = finding.get(
            "category", finding.get("finding_category", "Possible concern"),
        )
        status = finding.get("status", "unknown")

        with st.expander(
            f"{index}. {category.replace('_', ' ').title()} — {status.upper()}",
            expanded=True,
        ):
            observation = finding.get("observation", "")
            if observation:
                st.write("**What was noticed**")
                st.write(observation)

            st.write("**Suggested next step**")
            st.write(
                finding.get(
                    "recommendation",
                    "No suggested next step was saved.",
                )
            )

            evidence = finding.get("evidence", {})
            if evidence:
                with st.expander("Supporting details"):
                    st.json(evidence)

            limitation = finding.get("limitation", "")
            if limitation:
                st.caption(f"Important limitation: {limitation}")
else:
    if report.get("reasoning_error"):
        st.info("There is no finished risk check for this photo.")
    else:
        st.info("No possible safety concerns were recorded for this image.")

st.divider()

st.header("Objects detected")

if detections:
    detection_rows = []

    for item in detections:
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        detection_rows.append({
            "AI model": item.get("detector", "Not recorded"),
            "Object found": item.get("class_name", "Unknown"),
            "Confidence": f"{confidence * 100:.1f}%",
            "Image location": item.get("box_xyxy"),
        })

    st.write(
        "These are the objects the AI identified. "
        "Confidence is the model's score for a detection; it is not proof "
        "that the detection is correct."
    )

    st.dataframe(
        detection_rows,
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No object detection details were saved for this run.")

st.divider()

st.header("Which object checks ran?")

coverage = outcome.get("detector_coverage", {})

if coverage:
    st.write(
        "This section shows which object detectors contributed information "
        "to the screening. Limited detector coverage can affect the result."
    )
    st.json(coverage)
else:
    st.info("Detector coverage details were not saved for this run.")

st.header("Limits of this result")

st.write(
    outcome.get(
        "interpretation_limit",
        "This result is based on objects detected in one image. "
        "It cannot confirm every site condition or prove that the site is safe.",
    )
)

st.divider()

st.download_button(
    label="Download full screening report (JSON)",
    data=report_bytes,
    file_name=f"{run_dir.name}_report.json",
    mime="application/json",
    width="stretch",
)
