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


st.title("Context-aware vision reasoning for construction safety")
st.markdown(
    "### Master's capstone research workspace"
)
st.write(
    "This project investigates whether explicit image-space relationships, "
    "scene graphs, and cautious safety rules can make construction-image "
    "screening more traceable than object detection alone."
)

st.warning(
    "This is a research prototype, not a safety-certification or "
    "regulatory-compliance system. A photograph cannot establish actual "
    "machine operation, physical distance, or every site condition.",
    icon=":material/health_and_safety:",
)

with st.container(horizontal=True):
    st.metric("Visual model families", "3", help="PPE, toolset, and vehicles.")
    st.metric(
        "Implemented screening categories",
        "2",
        help=(
            "Potential missing-PPE detections and image-space "
            "worker–machinery relationships."
        ),
    )
    st.metric(
        "Integrated-system test set",
        "Not established",
        help=(
            "Independent scenario annotations are required before "
            "performance can be reported."
        ),
    )

st.header("Research questions")
questions = [
    (
        "RQ1 · Visual perception and relationships",
        "How accurately can computer vision identify relevant construction "
        "objects and extract spatial relationships that contribute to safety "
        "assessment?",
        "Existing detector checkpoints and reported baselines are available. "
        "A reproducible, machine-readable detector evaluation remains to be "
        "completed.",
    ),
    (
        "RQ2 · Context-aware reasoning",
        "Does incorporating spatial relationships, scene graphs, and "
        "contextual safety rules improve hazard-assessment performance over "
        "simpler detection-based approaches?",
        "A graph-linked rule prototype exists. This question is unanswered "
        "until both approaches are evaluated against the same independent "
        "scenario labels.",
    ),
    (
        "RQ3 · Explanation and recommendation quality",
        "Can the system produce evidence-grounded explanations and relevant "
        "preventive recommendations while communicating uncertainty?",
        "Outputs preserve detection and rule evidence. A blinded reviewer "
        "study using the proposal's explanation and recommendation rubric is "
        "still required.",
    ),
    (
        "RQ4 · Component contribution",
        "What contributions do spatial reasoning, scene graphs, and "
        "uncertainty handling make?",
        "The proposal makes this question conditional. No ablation result is "
        "claimed; controlled variants and paired evaluation are future work.",
    ),
]
for title, question, status in questions:
    with st.container(border=True):
        st.subheader(title)
        st.write(question)
        st.caption(status)

st.header("Reproducible detector validation")
validation_records = load_validation_records()
if validation_records:
    st.caption(
        "Latest recorded validation-split evaluations. These are not held-out "
        "test results and do not evaluate the contextual reasoning layer."
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
                "Evaluation date (UTC)": record["created_at_utc"],
            }
            for record in validation_records
        ],
        hide_index=True,
        width="stretch",
        alt="Reproducible validation metrics with checkpoint provenance",
    )
else:
    st.info(
        "No machine-readable validation evaluation has been recorded yet. "
        "Use the validation-only baseline commands in the README."
    )

st.header("Research progress")
phase_rows = [
    {
        "Phase": "1 · Research foundation",
        "Status": "In progress",
        "Evidence": "Advisor proposal supplied; focused literature review and novelty claim remain open.",
        "Next gate": "Document related work and finalize research questions with advisor.",
    },
    {
        "Phase": "2 · Dataset audit",
        "Status": "In progress",
        "Evidence": "Three detector datasets and a PPE annotation validator are present.",
        "Next gate": "Resolve provenance, licensing, split grouping, class mapping, and annotation completeness.",
    },
    {
        "Phase": "3 · Visual-perception baseline",
        "Status": "Baseline exists; reproduction pending",
        "Evidence": "Selected checkpoints, training arguments, run logs, and a written held-out report exist.",
        "Next gate": "Run the documented evaluation protocol and preserve machine-readable per-class results.",
    },
    {
        "Phase": "4 · Contextual reasoning",
        "Status": "Prototype implemented; evaluation pending",
        "Evidence": "Scene graph, image-space rules, evidence links, uncertainty reasons, and unit tests.",
        "Next gate": "Freeze rules on validation scenarios; assess relationship and finding quality independently.",
    },
    {
        "Phase": "5 · Prevention and application",
        "Status": "Prototype implemented; quality study pending",
        "Evidence": "Image screening, review, rule explanations, and cautious recommendation workflow.",
        "Next gate": "Use a blinded rubric to assess evidence, uncertainty, relevance, and unsupported claims.",
    },
    {
        "Phase": "6 · Experimental evaluation",
        "Status": "Protocol drafted; data pending",
        "Evidence": "Independent scenario evaluation schema and metric code are available.",
        "Next gate": "Collect independent labels, keep test data locked, and run controlled comparisons.",
    },
    {
        "Phase": "7 · Final documentation",
        "Status": "In progress",
        "Evidence": "Setup notes and evaluation protocol are documented.",
        "Next gate": "Prepare literature review, results, limitations, reproducibility record, report, and presentation.",
    },
]
st.dataframe(
    phase_rows,
    hide_index=True,
    width="stretch",
    alt="Capstone phases, evidence, status, and next completion gates",
)

st.header("Current system boundary")
st.markdown(
    """
- **In scope now:** PPE and person/machinery detections, image-space box
  relationships, traceable rule-triggered screening, uncertainty, and
  reviewer handoff.
- **Not implemented as assessed hazards:** fall protection, guardrails,
  excavation protection, rebar exposure, operational machine state, and
  regulatory compliance.
- **Not yet demonstrated:** integrated-system accuracy, causal benefit from
  scene graphs, calibrated risk, recommendation quality, and ablation effects.
- **Data integrity:** existing detector test splits, model weights, audit
  records, and manual reviews remain preserved. The 40-image duplicate audit
  is descriptive and is not an integrated-system test set.
"""
)

st.subheader("Suggested next research actions")
st.markdown(
    """
1. Finalize dataset-use permissions and document dataset provenance.
2. Freeze supported categories and the scenario annotation guide with the advisor.
3. Annotate an independent scenario set with explicit negative and uncertain labels.
4. Reproduce detector baselines with machine-readable metrics and checkpoint hashes.
5. Run paired baseline, rule-only, and graph-linked experiments; report error
   analysis and reviewer agreement without tuning on test examples.
"""
)

st.caption(
    "Project status is intentionally conservative: implemented software is "
    "not treated as experimental evidence."
)
