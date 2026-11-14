import json
from pathlib import Path

import streamlit as st

from construction_safety.config import EVALUATION_ROOT, PROJECT_ROOT


def latest_validation_records():
    if not EVALUATION_ROOT.is_dir():
        return []
    latest = {}
    for path in EVALUATION_ROOT.glob("*/evaluation.json"):
        with path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        if record.get("evaluation_split") != "val":
            continue
        detector = record.get("detector")
        previous = latest.get(detector)
        if previous is None or record.get("created_at_utc", "") > previous[
            "created_at_utc"
        ]:
            latest[detector] = record
    return list(latest.values())


def dataset_audit_records():
    audit_root = PROJECT_ROOT / "reports" / "dataset_audits"
    records = []
    for model_name in ("ppe", "toolset", "vehicles"):
        path = audit_root / f"{model_name}_dataset_audit.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as file:
                records.append(json.load(file))
    return records


st.title("Context-aware vision reasoning for construction safety")
st.markdown("### Master's capstone research workspace")
st.write(
    "This capstone asks whether explicit image-space relationships, scene "
    "graphs, and safety rules improve the traceability of construction-image "
    "screening compared with object detection alone."
)
st.warning(
    "A model prediction or image-space relationship is not proof of PPE "
    "absence, a safety violation, actual machine operation, physical distance, "
    "or an unsafe work condition.",
    icon=":material/health_and_safety:",
)

with st.container(horizontal=True):
    st.metric("Detector families", "3", help="PPE, toolset, and vehicles.")
    st.metric(
        "Implemented screening categories",
        "2",
        help=(
            "Potential missing-PPE detections and image-space "
            "person/machinery relationships."
        ),
    )
    st.metric(
        "Independent integrated test set",
        "Not established",
        help="Scenario labels are required before integrated performance claims.",
    )

st.header("Research questions and current evidence")
questions = [
    (
        "RQ1 · Perception and relationships",
        "How accurately can computer vision identify relevant objects and "
        "extract relationships that contribute to safety screening?",
        "Hashed validation records are available for all three detectors. "
        "Relationship quality still needs annotated reference relationships.",
    ),
    (
        "RQ2 · Context-aware reasoning",
        "Does structured contextual reasoning improve hazard-screening "
        "performance over simpler detection-based approaches?",
        "A graph-linked rule prototype exists. This question cannot be "
        "answered until paired approaches are evaluated against independent "
        "scenario labels.",
    ),
    (
        "RQ3 · Explanation quality",
        "Can explanations and recommendations be evidence-grounded and "
        "communicate uncertainty?",
        "The prototype preserves detections, boxes, rule IDs, limitations, and "
        "uncertainty. A blinded reviewer study remains outstanding.",
    ),
    (
        "RQ4 · Component contribution",
        "What contribution do spatial reasoning, scene graphs, and uncertainty "
        "handling make?",
        "Conditional in the proposal. Requires frozen, paired ablations and "
        "sufficiently independent annotations.",
    ),
]
for title, question, status in questions:
    with st.container(border=True):
        st.subheader(title)
        st.write(question)
        st.caption(status)

st.header("Latest reproducible detector validation")
records = latest_validation_records()
if records:
    st.caption(
        "Validation-split results only. These are not test-set results and "
        "do not establish contextual reasoning performance."
    )
    st.dataframe(
        [
            {
                "Detector": record["detector"],
                "Split": record["evaluation_split"],
                "Precision": record["metrics"].get("metrics/precision(B)"),
                "Recall": record["metrics"].get("metrics/recall(B)"),
                "mAP@50": record["metrics"].get("metrics/mAP50(B)"),
                "mAP@50–95": record["metrics"].get(
                    "metrics/mAP50-95(B)"
                ),
                "Checkpoint SHA-256": record["artifacts"][
                    "checkpoint_sha256"
                ][:12],
                "Run (UTC)": record["created_at_utc"],
            }
            for record in records
        ],
        hide_index=True,
        width="stretch",
        alt="Validation metrics with checkpoint provenance",
    )
else:
    st.info(
        "No evaluation record yet. Run the validation commands documented "
        "in the project README."
    )

st.header("Dataset integrity audit")
audits = dataset_audit_records()
if audits:
    st.dataframe(
        [
            {
                "Dataset": Path(record["dataset_yaml"]).parent.name,
                "Split": split_name,
                "Images": split["image_count"],
                "Annotations": split["annotation_count"],
                "Images without label file": split[
                    "images_without_label_file"
                ],
                "Orphan labels": split["orphan_label_file_count"],
                "Empty label files": split["empty_label_file_count"],
                "Structural errors": record["error_count"],
            }
            for record in audits
            for split_name, split in record["splits"].items()
        ],
        hide_index=True,
        width="stretch",
        alt="Dataset image and label audit by detector and split",
    )
    st.caption(
        "Missing label files require semantic review; they are not "
        "automatically treated as background negatives."
    )
else:
    st.info("Dataset audits have not yet been run in this clean project.")

st.header("Research phase gates")
phases = [
    {
        "Phase": "1 · Research foundation",
        "Status": "Preliminary targeted review drafted; broader review pending",
        "Acceptance gate": "Evidence-backed gap and advisor-confirmed questions.",
    },
    {
        "Phase": "2 · Dataset audit",
        "Status": "Data present; provenance and permitted use need confirmation",
        "Acceptance gate": "Document license, labels, class mapping, split and duplicate grouping.",
    },
    {
        "Phase": "3 · Detector baselines",
        "Status": "Validation records exist for all three detectors; final test locked",
        "Acceptance gate": "Hashed validation results and a single preregistered final test.",
    },
    {
        "Phase": "4 · Contextual reasoning",
        "Status": "Prototype and unit tests implemented",
        "Acceptance gate": "Freeze rules on validation scenarios; assess relationship and finding quality.",
    },
    {
        "Phase": "5 · Application and prevention",
        "Status": "Screening workflow implemented",
        "Acceptance gate": "Blinded, rubric-based evidence and recommendation review.",
    },
    {
        "Phase": "6 · Integrated evaluation",
        "Status": "Protocol and evaluator implemented; independent labels pending",
        "Acceptance gate": "Paired baseline and contextual comparisons on frozen scenarios.",
    },
    {
        "Phase": "7 · Capstone report",
        "Status": "Outline and method records established",
        "Acceptance gate": "Report actual results, error analysis, limitations, and reproducibility artifacts.",
    },
]
st.dataframe(
    phases,
    hide_index=True,
    width="stretch",
    alt="Capstone phases, evidence, and completion gates",
)

st.header("Scope and research integrity")
st.markdown(
    """
- **Prototype scope:** PPE, toolset, and vehicle detection; image-space
  bounding-box relationships; two cautious screening categories; human review.
- **Not assessed:** fall protection, guardrails, excavation protection, rebar,
  machine operating state, true physical distances, site exclusion zones,
  regulatory compliance, and injury probability.
- **Unanswered:** integrated accuracy, superiority over baselines, causal
  ablation effects, calibrated risk, and recommendation quality.
- **Data governance:** existing data, model weights, reports, and prior reviews
  are preserved outside this clean project folder. Dataset metadata indicates
  a private PPE license; verify rights before public distribution.
"""
)
