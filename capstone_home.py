
import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
EVALUATION_ROOT = ROOT / "reports" / "detector_evaluations"


def load_validation_records():
    if not EVALUATION_ROOT.is_dir():
        return []

    latest_by_detector = {}

    for report_path in EVALUATION_ROOT.glob("*/evaluation.json"):
        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        if report.get("evaluation_split") != "val":
            continue

        detector = report.get("detector")
        if detector not in {"ppe", "toolset", "vehicles"}:
            continue

        current = latest_by_detector.get(detector)

        if current is None or report.get("created_at_utc", "") > current[
            "created_at_utc"
        ]:
            latest_by_detector[detector] = report

    return list(latest_by_detector.values())


st.title("Context-aware vision reasoning for proactive construction safety")

st.write(
    "This research application helps identify possible safety concerns "
    "in construction-site pictures. It looks for workers, personal "
    "protective equipment, tools, vehicles, and machinery. It also "
    "examines image-space relationships between detected objects to "
    "identify situations that may need attention."
)

st.warning(
    "This application provides potential safety alerts, not final safety "
    "decisions. A warning does not automatically mean that a safety "
    "violation has occurred. Findings must be interpreted in context "
    "and checked against actual site conditions.",
    icon=":material/health_and_safety:",
)

st.header("What does the application check?")

st.markdown(
    """
- **Personal protective equipment:** Detects helmets, safety vests,
  gloves, boots, goggles, and supported missing-PPE predictions.
- **Tools and equipment:** Identifies supported construction tools
  and equipment.
- **Vehicles and machinery:** Identifies construction vehicles
  and machinery.
- **Worker and machinery relationships:** Examines image-space
  positions and proximity between detected workers and machinery.
- **Potential safety concerns:** Generates alerts when supported
  detection and relationship evidence meets the screening rules.
"""
)

st.header("Understanding the results")

st.markdown(
    """
- **High-confidence alert:** Strong detector evidence supports a
  potential concern. This is not proof of a safety violation.
- **Potential concern:** The available image evidence suggests a
  situation that may need attention.
- **Insufficient information:** The image does not provide enough
  evidence for a useful assessment.
- **No supported concern found:** The system did not identify a
  concern covered by its current screening rules. This does not
  guarantee that the site is safe.
"""
)

st.header("What should you do next?")

st.write(
    "Review the image and the generated findings. Check the actual "
    "worksite conditions before taking action. Human verification "
    "is available when needed, but it is not required for the system "
    "to generate potential alerts."
)

st.header("Project progress")

progress_rows = [
    {
        "Area": "Worker and PPE detection",
        "Status": "Implemented",
    },
    {
        "Area": "Tool and equipment detection",
        "Status": "Implemented",
    },
    {
        "Area": "Vehicle and machinery detection",
        "Status": "Implemented",
    },
    {
        "Area": "Cross-dataset object association",
        "Status": "Implemented; evaluation ongoing",
    },
    {
        "Area": "Worker and machinery relationship analysis",
        "Status": "Implemented; evaluation ongoing",
    },
    {
        "Area": "Automatic potential safety alerts",
        "Status": "Implemented; evaluation ongoing",
    },
    {
        "Area": "Safety explanations and recommendations",
        "Status": "Implemented; evaluation ongoing",
    },
    {
        "Area": "Model comparison and evaluation",
        "Status": "Ongoing",
    },
]

st.dataframe(
    progress_rows,
    hide_index=True,
    width="stretch",
)

st.caption(
    "Research prototype only. Image-space relationships do not establish "
    "physical distance, actual equipment operation, or confirmed safety "
    "violations. Findings should be checked against real site conditions."
)
