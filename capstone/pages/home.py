
import json
from pathlib import Path

import streamlit as st

from construction_safety.config import (
    EVALUATION_ROOT,
    MODEL_SPECS,
    PROJECT_ROOT,
)


def latest_validation_records():
    if not EVALUATION_ROOT.is_dir():
        return []

    latest = {}

    for path in EVALUATION_ROOT.glob("*/evaluation.json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                record = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        if record.get("evaluation_split") != "val":
            continue

        detector = record.get("detector")
        if not detector:
            continue

        previous = latest.get(detector)

        if (
            previous is None
            or record.get("created_at_utc", "")
            > previous.get("created_at_utc", "")
        ):
            latest[detector] = record

    return list(latest.values())


def dataset_audit_records():
    audit_root = PROJECT_ROOT / "reports" / "dataset_audits"
    records = []

    for model_name in ("ppe", "toolset35", "vehicles"):
        path = audit_root / f"{model_name}_dataset_audit.json"

        if not path.is_file():
            continue

        try:
            with path.open("r", encoding="utf-8") as file:
                records.append(json.load(file))
        except (OSError, json.JSONDecodeError):
            continue

    return records


st.title("ConstructionSafetyAI")
st.caption("Construction Site Safety Screening")

st.write(
    """
    Upload a construction site photo to check for workers, safety equipment,
    tools, and vehicles.

    The app uses AI models to find objects, look for possible safety concerns,
    and create a report with the results.
    """
)

st.warning(
    "This app helps identify possible safety concerns in photos. "
    "It cannot confirm whether a site is safe or unsafe. "
    "It does not replace an actual site inspection.",
    icon=":material/health_and_safety:",
)

st.header("What this app does")

st.markdown(
    """
    **1. Finds objects**

    Looks for people, safety equipment, tools, and construction vehicles.

    **2. Checks repeated detections**

    Different models may find the same object. The app checks for repeated
    results and keeps the strongest prediction in the final list.

    **3. Looks at the image**

    Checks which objects appear near each other in the photo.

    **4. Finds possible safety concerns**

    Looks for possible missing safety equipment and workers who appear
    close to machinery in the image.

    **5. Creates a report**

    Shows marked-up photos, detected objects, possible concerns, and
    screening results automatically.
    """
)

st.header("Available detection systems")

detector_labels = [
    spec.get("label", name)
    for name, spec in MODEL_SPECS.items()
    if name in ("ppe", "toolset35", "vehicles")
]

if detector_labels:
    columns = st.columns(len(detector_labels))

    for column, label in zip(columns, detector_labels):
        with column:
            st.metric("Detection system", label)

st.caption(
    "The app supports two model choices: YOLO11n and RT-DETR-L."
)

st.header("How to understand the results")

with st.container(border=True):
    st.subheader("Detected objects")
    st.write(
        "The AI model found an object that may be a person, piece of "
        "equipment, tool, or vehicle. The confidence score shows how "
        "sure the model is about its prediction."
    )

with st.container(border=True):
    st.subheader("Repeated detections")
    st.write(
        "More than one model may find the same object. The app keeps "
        "the strongest result in the final list and saves the original "
        "predictions in the report."
    )

with st.container(border=True):
    st.subheader("Possible missing safety equipment")
    st.write(
        "The app may detect signs that a worker is missing a helmet, "
        "vest, gloves, boots, or goggles. A photo cannot always show "
        "whether the worker is actually wearing the equipment."
    )

with st.container(border=True):
    st.subheader("Workers and machinery")
    st.write(
        "The app checks how close a worker and a machine appear in the "
        "photo. This does not tell us their actual distance or prove "
        "that the machine is dangerous."
    )

with st.container(border=True):
    st.subheader("Overall results")
    st.markdown(
        """
        - **High-confidence alert:** A result meets the app's alert rules.
        - **Possible concern:** The app found something that needs checking.
        - **Not enough information:** The app could not check everything.
        - **No supported concerns found:** The app found no concerns covered
          by its current checks. This does not mean the site is completely safe.
        """
    )

st.header("Model test results")

records = latest_validation_records()

if records:
    st.caption(
        "These results show how well each detection model performed "
        "on its test images. They do not measure the accuracy of the "
        "entire safety screening system."
    )

    rows = []

    for record in records:
        metrics = record.get("metrics", {})
        artifacts = record.get("artifacts", {})

        rows.append(
            {
                "Model": record.get("detector", "Unknown"),
                "Test group": record.get("evaluation_split", "Unknown"),
                "Precision": metrics.get("metrics/precision(B)"),
                "Recall": metrics.get("metrics/recall(B)"),
                "mAP at 50%": metrics.get("metrics/mAP50(B)"),
                "mAP at 50–95%": metrics.get("metrics/mAP50-95(B)"),
                "Model file ID": (
                    artifacts.get("checkpoint_sha256", "")[:12]
                ),
                "Test date (UTC)": record.get("created_at_utc", ""),
            }
        )

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
    )
else:
    st.info("No model test results are available yet.")

st.header("Dataset checks")

audits = dataset_audit_records()

if audits:
    rows = []

    for record in audits:
        dataset_name = Path(
            record.get("dataset_yaml", "Unknown")
        ).parent.name

        for split_name, split in record.get("splits", {}).items():
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Data group": split_name,
                    "Images": split.get("image_count", 0),
                    "Label files": split.get("annotation_count", 0),
                    "Images missing label files": split.get(
                        "images_without_label_file", 0
                    ),
                    "Extra label files": split.get(
                        "orphan_label_file_count", 0
                    ),
                    "Empty label files": split.get(
                        "empty_label_file_count", 0
                    ),
                    "Data errors": record.get("error_count", 0),
                }
            )

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Images without label files need to be checked during data "
        "preparation. They are not automatically treated as background images."
    )
else:
    st.info("No dataset check results are available yet.")

st.header("What the app cannot do")

st.markdown(
    """
    - It cannot prove that a worker is missing safety equipment.
    - It cannot measure the real distance between a worker and a machine.
    - It cannot tell whether a machine is running.
    - It cannot confirm that a construction site follows safety regulations.
    - It does not check every possible construction hazard.
    - It cannot replace a trained person's inspection of the actual site.
    """
)
