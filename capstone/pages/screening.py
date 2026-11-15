
import json
import tempfile
from pathlib import Path

import streamlit as st

from construction_safety.inference import run_inference, save_run
from construction_safety.review import (
    get_automatic_review_queue,
    load_review,
    save_review,
)


st.title("Construction image screening")
st.write(
    "Run the selected local detectors, inspect evidence-linked relationships, "
    "and export a structured screening report."
)
st.warning(
    "All outputs are research screening candidates. A qualified person must "
    "verify them against the image and actual site conditions.",
    icon=":material/health_and_safety:",
)

uploaded_image = st.file_uploader(
    "Select a construction-site image",
    type=["jpg", "jpeg", "png", "webp"],
)

with st.form("inference_settings"):
    model_names = st.multiselect(
        "Detector models",
        options=["ppe", "toolset", "vehicles", "toolset35"],
        default=["ppe", "toolset", "vehicles"],
        format_func={
            "ppe": "PPE",
            "toolset": "Toolset",
            "vehicles": "Vehicles",
            "toolset35": "Toolset-35 (35 classes)",
        }.get,
    )

    confidence = st.slider(
        "PPE, Toolset, and Toolset-35 confidence threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    vehicle_confidence = st.slider(
        "Vehicle confidence threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help=(
            "Applies to the nine non-Roller vehicle classes. "
            "Roller uses a separate provisional confidence threshold of 0.70. "
            "Higher thresholds can reduce false positives but may miss real vehicles."
        ),
    )

    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640],
        value=640,
    )

    submitted = st.form_submit_button(
        "Run selected detectors",
        type="primary",
        disabled=uploaded_image is None,
    )

if submitted:
    if not model_names:
        st.error("Select at least one detector.")
        st.stop()

    try:
        with tempfile.TemporaryDirectory(
            prefix="construction-capstone-input-"
        ) as temp_dir:
            image_path = Path(temp_dir) / Path(uploaded_image.name).name
            image_path.write_bytes(uploaded_image.getvalue())

            with st.spinner("Running local detectors on CPU..."):
                annotated, report = run_inference(
                    image_path,
                    confidence=confidence,
                    image_size=image_size,
                    model_names=tuple(model_names),
                    confidence_by_model={"vehicles": vehicle_confidence},
                    confidence_by_class={"vehicles": {"Roller": 0.70}},
                )

        run_folder, report_path = save_run(report)
        annotated_path = run_folder / "annotated_detections.jpg"
        annotated.save(annotated_path, quality=92)

        st.session_state["screening_result"] = {
            "report": report,
            "annotated_path": str(annotated_path),
            "report_path": str(report_path),
            "run_folder": str(run_folder),
        }

    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Image screening failed: {error}")
        st.stop()


result = st.session_state.get("screening_result")

if result:
    report = result["report"]
    findings = report["safety_assessment"]["findings"]

    try:
        review, review_path = load_review(
            report,
            result["report_path"],
        )
        current_queue = get_automatic_review_queue(report)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not load human review: {error}")
        st.stop()

    # Preserve existing decisions only for findings automatically queued.
    previous_decisions = {
        row["finding_id"]: row
        for row in review["review_decisions"]
        if row["finding_id"] in current_queue
    }

    # Legacy v1 records may not have a decision for every automatically
    # queued finding, so initialize any missing decisions as Pending.
    for finding_id in current_queue:
        previous_decisions.setdefault(
            finding_id,
            {
                "finding_id": finding_id,
                "decision": "Pending",
                "reviewer_notes": "",
            },
        )

    triage = report["safety_assessment"].get("review_triage", {})
    disposition_counts = triage.get("disposition_counts", {})

    st.header("Screening results")

    with st.container(horizontal=True):
        st.metric("Detections", len(report["detections"]))
        st.metric("Potential findings", len(findings))
        st.metric("Automatically queued", len(current_queue))

    if disposition_counts:
        st.caption(
            "Automatic triage: "
            f"{disposition_counts.get('review_required', 0)} require review · "
            f"{disposition_counts.get('insufficient_evidence', 0)} "
            "have insufficient evidence · "
            f"{disposition_counts.get('no_review_required', 0)} "
            "do not require review under the implemented rules."
        )

    image_column, finding_column = st.columns([1.2, 1])

    image_column.image(
        result["annotated_path"],
        caption="Detector predictions over the uploaded image",
        width="stretch",
        alt="Construction image with detector boxes and class labels",
    )

    with finding_column:
        st.subheader("Rule-triggered findings")

        if not findings:
            st.info(
                "No supported rule was triggered. This does not establish "
                "that the image or site is hazard-free."
            )

        for finding in findings:
            finding_id = finding["finding_id"]
            disposition = finding.get("review_disposition")

            if disposition == "review_required":
                disposition_label = "Automatically queued for human review"
            elif disposition == "no_review_required":
                disposition_label = (
                    "No human review required under this rule"
                )
            elif disposition == "insufficient_evidence":
                disposition_label = (
                    "Insufficient evidence — automatically queued"
                )
            else:
                disposition_label = (
                    "Unknown disposition — conservatively queued"
                )

            with st.container(border=True):
                st.markdown(
                    f"**{finding['finding_category']}** · "
                    f"`{finding['rule_id']}`"
                )
                st.write(finding["observation"])
                st.write(finding["recommendation"])
                st.caption(finding["limitation"])
                st.caption(
                    "Uncertainty: "
                    + "; ".join(finding["uncertainty_reasons"])
                )
                st.caption(disposition_label)

                if finding_id in current_queue:
                    st.caption(
                        "Automatic triage reason: "
                        + finding.get(
                            "review_reason",
                            "Human verification is required.",
                        )
                    )

    if findings:
        st.header("Automatic human review queue")
        st.caption(
            "The system determines which findings enter this queue. "
            "You cannot manually add or remove findings. Reviewer decisions "
            "are recorded separately from model predictions."
        )

        if current_queue:
            st.metric("Findings requiring review", len(current_queue))

            st.subheader("Review queued findings")

            review_decisions = []

            with st.form("finding_review_form"):
                for finding_id in current_queue:
                    finding = next(
                        item for item in findings
                        if item["finding_id"] == finding_id
                    )

                    previous = previous_decisions[finding_id]

                    decision_options = [
                        "Pending",
                        "Confirmed",
                        "Rejected",
                        "Uncertain",
                    ]

                    with st.container(border=True):
                        st.markdown(
                            f"**{finding_id}** · "
                            f"{finding['finding_category']}"
                        )
                        st.write(finding["observation"])
                        st.caption(
                            finding.get(
                                "review_reason",
                                "Human verification is required.",
                            )
                        )

                        decision = st.selectbox(
                            "Reviewer decision",
                            decision_options,
                            index=decision_options.index(
                                previous["decision"]
                            ),
                            key=f"review_decision_{finding_id}",
                        )

                        notes = st.text_area(
                            "Reviewer notes",
                            value=previous.get("reviewer_notes", ""),
                            key=f"review_notes_{finding_id}",
                        )

                    review_decisions.append({
                        "finding_id": finding_id,
                        "decision": decision,
                        "reviewer_notes": notes.strip(),
                    })

                review_submitted = st.form_submit_button(
                    "Save human review",
                    type="primary",
                )

            if review_submitted:
                try:
                    save_review(
                        report,
                        review_path,
                        review_decisions,
                    )
                except (OSError, ValueError) as error:
                    st.error(f"Could not save human review: {error}")
                    st.stop()

                st.success(
                    "Human decisions saved separately from model output."
                )
                st.rerun()
        else:
            st.info(
                "The automatic review queue is empty. No current rule "
                "requires review for these findings. This does not mean "
                "the site or image is confirmed safe."
            )

    with st.expander("Saved review record"):
        st.json(review)

    # Display possible duplicate physical objects separately from
    # safety relationships and findings.
    with st.expander("Possible duplicate detections"):
        scene_graph = report.get("scene_graph", {})
        associations = scene_graph.get("object_associations", [])

        nodes_by_id = {
            node["object_id"]: node
            for node in scene_graph.get("nodes", [])
        }

        if associations:
            st.caption(
                "These are candidate associations based on bounding-box "
                "overlap across detectors. They do not confirm physical "
                "object identity."
            )

            for association in associations:
                st.markdown(
                    f"**{association['association_id']}**"
                )

                member_ids = association.get("member_ids", [])

                for member_id in member_ids:
                    node = nodes_by_id.get(member_id, {})

                    st.write({
                        "Object ID": member_id,
                        "Detector": node.get("detector"),
                        "Class": node.get("class_name"),
                        "Confidence": node.get("confidence"),
                        "Bounding box": node.get("bbox_xyxy"),
                    })

                evidence = association.get("evidence", {})

                st.write(
                    "Intersection over union:",
                    evidence.get("intersection_over_union"),
                )
                st.write(
                    "Matching threshold:",
                    evidence.get("threshold"),
                )

                st.caption(
                    association.get("interpretation_limit", "")
                )
                st.divider()
        else:
            st.info(
                "No possible cross-detector duplicate detections found."
            )

    with st.expander("Scene graph and evidence"):
        st.json(report["scene_graph"])

    with st.expander("Unassessed conditions"):
        st.write(report["safety_assessment"]["unassessed_conditions"])
        st.write(report["safety_assessment"]["risk_assessment"]["reason"])

    st.download_button(
        "Download structured screening report",
        data=json.dumps(report, indent=2),
        file_name="construction_safety_screening.json",
        mime="application/json",
    )

    st.caption(f"Research run saved to `{result['run_folder']}`.")
