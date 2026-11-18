
import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "reports" / "inference_runs"


st.title("Inference reports")
st.caption("Saved results from unified construction safety screening.")


def find_reports():
    reports = []

    if not RUNS_DIR.exists():
        return reports

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        json_path = run_dir / "detections.json"

        if json_path.exists():
            reports.append((run_dir, json_path))

    return sorted(
        reports,
        key=lambda item: item[0].stat().st_mtime,
        reverse=True,
    )


reports = find_reports()

if not reports:
    st.warning("No saved inference reports were found.")
    st.caption(f"Expected folder: {RUNS_DIR}")
    st.stop()


selected_run = st.selectbox(
    "Select an inference run",
    reports,
    format_func=lambda item: item[0].name,
)

run_dir, json_path = selected_run

try:
    report_bytes = json_path.read_bytes()
    report = json.loads(report_bytes)
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Could not read this report: {error}")
    st.stop()


outcome = report.get("screening_outcome", {})
safety = report.get("safety_assessment", {})
findings = safety.get("findings", [])
detections = report.get("analysis_detections", [])

st.subheader("Screening summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Outcome",
    outcome.get("outcome", "Unavailable"),
)
col2.metric(
    "Potential findings",
    len(findings),
)
col3.metric(
    "Retained detections",
    report.get("retained_detection_count", len(detections)),
)
col4.metric(
    "Suppressed duplicates",
    report.get("suppressed_duplicate_count", 0),
)

st.caption(outcome.get("explanation", "No explanation available."))

st.divider()

st.subheader("Detection image")

image_path = run_dir / "unified_detections.jpg"

if image_path.exists():
    st.image(
        str(image_path),
        caption=f"Inference run: {run_dir.name}",
        use_container_width=True,
    )
else:
    st.info("No annotated detection image was saved for this run.")

st.divider()

st.subheader("Potential findings")

if findings:
    for index, finding in enumerate(findings, start=1):
        category = finding.get(
            "finding_category",
            "Uncategorized finding",
        )
        status = finding.get("status", "unknown")

        with st.expander(
            f"{index}. {category.replace('_', ' ').title()} — {status.upper()}",
            expanded=True,
        ):
            st.write(finding.get("observation", ""))
            st.write("**Recommendation**")
            st.write(finding.get("recommendation", "Not provided."))
            st.write("**Evidence**")
            st.json(finding.get("evidence", {}))
            st.caption(finding.get("limitation", ""))
else:
    st.info("No potential findings were recorded in this run.")

st.divider()

st.subheader("Detection details")

if detections:
    st.dataframe(
        [
            {
                "Detector": item.get("detector"),
                "Class": item.get("class_name"),
                "Confidence": round(
                    float(item.get("confidence", 0)), 4
                ),
                "Bounding box": item.get("box_xyxy"),
            }
            for item in detections
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No retained detection details are available.")

st.divider()

st.subheader("Detector coverage")
st.json(outcome.get("detector_coverage", {}))

st.subheader("Interpretation limitations")
st.write(
    outcome.get(
        "interpretation_limit",
        "No interpretation note was recorded.",
    )
)

st.download_button(
    label="Download complete JSON report",
    data=report_bytes,
    file_name=f"{run_dir.name}_report.json",
    mime="application/json",
    use_container_width=True,
)
