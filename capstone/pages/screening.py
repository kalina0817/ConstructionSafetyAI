
import json
import tempfile
from pathlib import Path

import streamlit as st

from construction_safety.inference import run_inference, save_run


st.title("Check a construction site image")

st.write(
    "Run the selected local detectors to analyze a construction-site image. "
    "The system automatically processes detections, evaluates supported "
    "safety conditions, and generates a structured screening report."
)

st.warning(
    "This is an automated research screening system, not a safety "
    "certification system. Image-based findings and proximity estimates "
    "do not establish actual site conditions.",
    icon=":material/info:",
)


uploaded_image = st.file_uploader(
    "Upload a construction site photo",
    type=["jpg", "jpeg", "png", "webp"],
)


with st.form("inference_settings"):
    model_names = st.multiselect(
        "What should the app look for?",
        options=["ppe", "toolset35", "vehicles"],
        default=["ppe", "toolset35", "vehicles"],
        format_func={
            "ppe": "PPE",
            "toolset35": "Toolset-35 (35 classes)",
            "vehicles": "Vehicles",
        }.get,
    )

    variant_options = {
        "YOLO11n": "yolo11n",
        "RT-DETR-L": "rtdetr_l",
    }

    model_variants = {}

    st.markdown("**Choose a model for each category**")

    if "ppe" in model_names:
        model_variants["ppe"] = variant_options[
            st.selectbox(
                "Safety equipment model",
                options=list(variant_options),
                index=0,
                key="screening_ppe_variant",
            )
        ]

    if "toolset35" in model_names:
        model_variants["toolset35"] = variant_options[
            st.selectbox(
                "Toolset-35 model",
                options=list(variant_options),
                index=0,
                key="screening_toolset35_variant",
            )
        ]

    if "vehicles" in model_names:
        model_variants["vehicles"] = variant_options[
            st.selectbox(
                "Vehicle detection model",
                options=list(variant_options),
                index=0,
                key="screening_vehicles_variant",
            )
        ]

    confidence = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    vehicle_confidence = st.slider(
        "Vehicle detection confidence",
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
        "Image processing size",
        options=[320, 416, 512, 640],
        value=640,
    )

    submitted = st.form_submit_button(
        "Check this image",
        type="primary",
        disabled=uploaded_image is None,
    )


if submitted:
    if not model_names:
        st.error("Please choose at least one category to check.")
        st.stop()

    try:
        with tempfile.TemporaryDirectory(
            prefix="construction-capstone-input-"
        ) as temp_dir:
            image_path = Path(temp_dir) / Path(uploaded_image.name).name
            image_path.write_bytes(uploaded_image.getvalue())

            with st.spinner(
                "Checking your image. This may take a little while..."
            ):
                annotated, report = run_inference(
                    image_path,
                    confidence=confidence,
                    image_size=image_size,
                    model_names=tuple(model_names),
                    model_variants=model_variants,
                    confidence_by_model={
                        "vehicles": vehicle_confidence,
                    },
                    confidence_by_class={
                        "vehicles": {"Roller": 0.70},
                    },
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
        st.error(f"The image could not be checked: {error}")
        st.stop()


result = st.session_state.get("screening_result")


if result:
    report = result["report"]
    safety_assessment = report.get("safety_assessment", {})
    findings = safety_assessment.get("findings", [])

    screening_outcome = report.get("screening_outcome", {})

    potential_alerts = [
        finding
        for finding in findings
        if finding.get("status") == "potential"
    ]

    raw_detections = report.get("raw_detections", [])
    final_detections = report.get("detections", [])

    raw_detections_by_id = {
        detection["detection_id"]: detection
        for detection in raw_detections
        if detection.get("detection_id")
    }

    same_detector_suppressions = [
        detection
        for detection in final_detections
        if detection.get("suppressed_detection_ids")
    ]

    st.header("Screening results")

    # ---------------------------------------------------------
    # Overall result
    # ---------------------------------------------------------

    st.subheader("Overall result")

    outcome_name = (
        screening_outcome.get("outcome")
        or screening_outcome.get("classification")
        or screening_outcome.get("status")
        or "UNKNOWN"
    )

    outcome_explanations = {
        "HIGH_CONFIDENCE_ALERT": (
            "A high-confidence alert was generated under validated "
            "screening criteria."
        ),
        "POTENTIAL": (
            "One or more supported screening rules generated potential "
            "findings. These findings are not confirmed incidents."
        ),
        "INSUFFICIENT": (
            "The selected detector coverage is insufficient to complete "
            "the configured screening checks."
        ),
        "NO_SUPPORTED": (
            "No findings were generated by the supported screening rules "
            "with the required detector coverage. This does not mean the "
            "site is hazard-free."
        ),
        "UNKNOWN": (
            "The overall result is not available."
        ),
    }

    outcome_labels = {
        "HIGH_CONFIDENCE_ALERT": "High-confidence alert",
        "POTENTIAL": "Possible safety concern",
        "INSUFFICIENT": "Not enough information",
        "NO_SUPPORTED": "No supported concerns found",
    }

    outcome_label = outcome_labels.get(outcome_name, outcome_name)

    outcome_explanation = screening_outcome.get(
        "explanation",
        outcome_explanations.get(
            outcome_name,
            "No outcome explanation is available.",
        ),
    )

    if outcome_name == "HIGH_CONFIDENCE_ALERT":
        st.error(f"**{outcome_label}**")
    elif outcome_name == "POTENTIAL":
        st.warning(f"**{outcome_label}**")
    elif outcome_name == "NO_SUPPORTED":
        st.success(f"**{outcome_label}**")
    elif outcome_name == "INSUFFICIENT":
        st.warning(f"**{outcome_label}**")
    else:
        st.info(f"**{outcome_label}**")

    st.write(outcome_explanation)

    outcome_limitation = (
        screening_outcome.get("limitation")
        or screening_outcome.get("interpretation_limit")
    )

    if outcome_limitation:
        st.caption(outcome_limitation)

    detector_coverage = screening_outcome.get(
        "detector_coverage",
        {},
    )

    if detector_coverage:
        with st.expander("What the app was able to check"):
            st.json(detector_coverage)

    if screening_outcome.get("high_confidence_alert_available") is False:
        high_confidence_limitation = (
            screening_outcome.get("high_confidence_alert_limitation")
            or (
                "High-confidence alert is unavailable because "
                "independently validated criteria have not been established."
            )
        )

        st.caption(high_confidence_limitation)

    # ---------------------------------------------------------
    # Detection metrics
    # ---------------------------------------------------------

    detection_summary = report.get("detection_summary", {})

    with st.container(horizontal=True):
        st.metric(
            "Initial detections",
            detection_summary.get(
                "raw_detection_count",
                len(raw_detections),
            ),
        )

        st.metric(
            "Final detections",
            detection_summary.get(
                "final_detection_count",
                len(final_detections),
            ),
        )

        st.metric(
            "Combined detections",
            detection_summary.get(
                "merged_prediction_count",
                0,
            ),
        )

        st.metric(
            "Possible safety concerns",
            len(potential_alerts),
        )

    # ---------------------------------------------------------
    # Annotated image and potential alerts
    # ---------------------------------------------------------

    image_column, alert_column = st.columns([1.2, 1])

    image_column.image(
        result["annotated_path"],
        caption="Photo with detected objects marked",
        width="stretch",
        alt="Construction image with detector boxes and class labels",
    )

    with alert_column:
        st.subheader("Possible safety concerns")

        if not potential_alerts:
            st.info(
                "No potential alerts were generated by the configured "
                "screening rules. This does not establish that the image "
                "or site is hazard-free."
            )

        for alert in potential_alerts:
            evidence = alert.get("evidence", {})
            class_name = str(
                evidence.get("class_name", "")
            ).lower()

            rule_id = alert.get("rule_id", "")

            if rule_id == "MISSING_PPE_PREDICTION_REQUIRES_REVIEW":
                equipment_names = {
                    "no_boots": "safety boots",
                    "no_helmet": "safety helmet",
                    "no_gloves": "protective gloves",
                    "no_goggle": "safety goggles",
                    "no_goggles": "safety goggles",
                    "no_vest": "high-visibility vest",
                }

                equipment = equipment_names.get(
                    class_name,
                    "required protective equipment",
                )

                alert_title = (
                    f"Possible safety concern PPE HAZARD: POSSIBLE MISSING "
                    f"{equipment.upper()}"
                )

            elif rule_id == "IMAGE_SPACE_PROXIMITY_REQUIRES_VERIFICATION":
                alert_title = (
                    "Possible safety concern WORKER–MACHINERY PROXIMITY ALERT"
                )

            else:
                alert_title = "Possible safety concern SAFETY HAZARD"

            with st.container(border=True):
                st.markdown(f"**{alert_title}**")

                st.write(
                    alert.get(
                        "observation",
                        "No observation available.",
                    )
                )

                st.markdown("**What to do next**")

                st.write(
                    alert.get(
                        "recommendation",
                        "No recommendation available.",
                    )
                )

                limitation = alert.get("limitation", "")

                if limitation:
                    st.caption(limitation)

                st.caption(
                    "App result: Possible safety concern"
                )

                # Display uncertainty reasons as plain text.
                # This avoids Markdown interpreting list markers
                # inside the reason strings.
                uncertainty_reasons = alert.get(
                    "uncertainty_reasons",
                    [],
                )

                if not isinstance(uncertainty_reasons, list):
                    uncertainty_reasons = []

                uncertainty_reasons = [
                    reason.strip()
                    for reason in uncertainty_reasons
                    if isinstance(reason, str) and reason.strip()
                ]

                if uncertainty_reasons:
                    with st.expander(
                        "Why this result may be uncertain"
                    ):
                        for index, reason in enumerate(
                            uncertainty_reasons,
                            start=1,
                        ):
                            st.text(f"{index}. {reason}")
                else:
                    st.caption(
                        "No separate uncertainty explanations were "
                        "provided for this finding."
                    )

    # ---------------------------------------------------------
    # Same-detector PPE duplicate suppression
    # ---------------------------------------------------------

    with st.expander(
        f"Repeated safety equipment detections ({len(same_detector_suppressions)})"
    ):
        if same_detector_suppressions:
            st.caption(
                "Overlapping predictions from the same PPE detector and "
                "the same missing-PPE class are grouped using bounding-box "
                "IoU. The highest-confidence prediction is retained. "
                "Suppressed predictions remain available in raw_detections "
                "in the structured report."
            )

            for retained in same_detector_suppressions:
                suppressed_ids = retained.get(
                    "suppressed_detection_ids",
                    [],
                )

                st.markdown(
                    f"**{retained.get('class_name', 'Unknown class')}**"
                )

                st.write({
                    "Retained detection ID": retained.get("detection_id"),
                    "Retained confidence": retained.get("confidence"),
                    "Detector": retained.get("detector"),
                    "Bounding box": retained.get("box_xyxy"),
                    "Possible duplicate count": retained.get(
                        "possible_duplicate_count",
                        1,
                    ),
                    "IoU suppression threshold": retained.get(
                        "same_detector_duplicate_iou_threshold"
                    ),
                })

                st.markdown("**Suppressed original predictions**")

                for detection_id in suppressed_ids:
                    original = raw_detections_by_id.get(
                        detection_id,
                        {},
                    )

                    st.write({
                        "Detection ID": detection_id,
                        "Detector": original.get(
                            "detector",
                            "Unavailable",
                        ),
                        "Class": original.get(
                            "class_name",
                            "Unavailable",
                        ),
                        "Confidence": original.get("confidence"),
                        "Bounding box": original.get("box_xyxy"),
                    })

                st.divider()

        else:
            st.info(
                "No same-detector missing-PPE duplicate predictions "
                "were suppressed in this run."
            )

    # ---------------------------------------------------------
    # Objects detected by more than one model
    # ---------------------------------------------------------

    with st.expander("Objects detected by more than one model"):
        raw_scene_graph = report.get("raw_scene_graph", {})

        associations = report.get(
            "possible_duplicate_associations",
            raw_scene_graph.get("object_associations", []),
        )

        nodes_by_id = {
            node["object_id"]: node
            for node in raw_scene_graph.get("nodes", [])
            if "object_id" in node
        }

        if associations:
            st.caption(
                "These are candidate associations based on bounding-box "
                "overlap across detectors. They do not confirm physical "
                "object identity. The highest-confidence prediction from "
                "each associated group is retained in the final detections."
            )

            for association in associations:
                association_id = association.get(
                    "association_id",
                    "Possible duplicate group",
                )

                st.markdown(f"**{association_id}**")

                member_ids = association.get("member_ids", [])

                for member_id in member_ids:
                    node = nodes_by_id.get(member_id, {})

                    st.write({
                        "Object ID": member_id,
                        "Detector": node.get("detector", "Unavailable"),
                        "Class": node.get("class_name", "Unavailable"),
                        "Confidence": node.get("confidence"),
                        "Bounding box": node.get("box_xyxy"),
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

    # ---------------------------------------------------------
    # Detected objects and their relationships
    # ---------------------------------------------------------

    with st.expander("Detected objects and their relationships"):
        st.json(report.get("scene_graph", {}))

    # ---------------------------------------------------------
    # Items the app could not check and risk assessment
    # ---------------------------------------------------------

    with st.expander("Items the app could not check"):
        unassessed_conditions = safety_assessment.get(
            "unassessed_conditions",
            [],
        )

        if unassessed_conditions:
            for condition in unassessed_conditions:
                st.write(f"- {condition}")
        else:
            st.write("No unassessed conditions were listed.")

        risk_assessment = safety_assessment.get(
            "risk_assessment",
            {},
        )

        if risk_assessment:
            st.write(risk_assessment.get("reason", ""))

            st.caption(
                "Risk scores and probabilities are not calculated "
                "because calibrated risk estimates are unavailable."
            )

    # ---------------------------------------------------------
    # Structured report download
    # ---------------------------------------------------------

    st.download_button(
        "Download screening report",
        data=json.dumps(report, indent=2),
        file_name="construction_safety_screening.json",
        mime="application/json",
    )

    st.caption(
        f"Results saved to `{result['run_folder']}`."
    )

