
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = ROOT / "unified_inference.py"
RUNS_DIR = ROOT / "reports" / "inference_runs"
UPLOAD_DIR = ROOT / "input_uploads"

CONTAINMENT_THRESHOLD = 0.80

RECOMMENDATION_RULES = {
    "no_vest": {
        "observation": "The PPE detector predicts a possible missing high-visibility-vest condition.",
        "recommendation": (
            "Review the worker's high-visibility clothing and verify "
            "whether it meets the applicable site requirements."
        ),
        "limitation": (
            "The image alone does not establish the site's required "
            "high-visibility clothing policy or prove a regulatory violation."
        )
    },
    "no_gloves": {
        "observation": "The PPE detector predicts a possible missing safety-gloves condition.",
        "recommendation": (
            "Review the worker's hand protection and determine whether "
            "task-appropriate gloves are required for the activity."
        ),
        "limitation": (
            "The image alone does not establish the task, glove requirements, "
            "or whether alternative hand protection is being used."
        )
    }
}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def build_adjudicated_report(run_folder):
    detections_file = run_folder / "detections.json"
    review_file = run_folder / "human_review.json"
    output_file = run_folder / "adjudicated_findings.json"

    detections_report = load_json(detections_file)
    review_report = load_json(review_file)

    review_source = Path(
        review_report.get("source_report", "")
    ).resolve()

    if review_source != detections_file.resolve():
        raise ValueError(
            "Human review references a different detection report."
        )

    decisions = review_report.get("review_decisions", [])

    confirmed = []
    rejected = []
    uncertain = []
    pending = []

    for item in decisions:
        decision = item.get("decision", "Pending")
        candidate = item.get("candidate", {})

        finding = {
            "candidate_index": item.get("candidate_index"),
            "candidate_type": candidate.get("candidate_type"),
            "class_name": candidate.get("class_name"),
            "detector": candidate.get("detector"),
            "confidence": candidate.get("confidence"),
            "person_class": candidate.get("person_class"),
            "machinery_class": candidate.get("machinery_class"),
            "normalized_image_distance": candidate.get(
                "normalized_image_distance"
            ),
            "human_decision": decision,
            "reviewer_notes": item.get("reviewer_notes", ""),
            "original_candidate": candidate
        }

        if decision == "Confirmed":
            confirmed.append(finding)
        elif decision == "Rejected":
            rejected.append(finding)
        elif decision == "Uncertain":
            uncertain.append(finding)
        else:
            pending.append(finding)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_detection_report": str(detections_file),
        "source_human_review": str(review_file),
        "total_reviewed_candidates": len(decisions),
        "summary": {
            "confirmed_predictions": len(confirmed),
            "rejected_predictions": len(rejected),
            "uncertain_predictions": len(uncertain),
            "pending_predictions": len(pending)
        },
        "confirmed_findings": confirmed,
        "rejected_predictions": rejected,
        "uncertain_findings": uncertain,
        "pending_findings": pending,
        "interpretation_note": (
            "Confirmed findings reflect human review of the visible image. "
            "They are not automatically proof of a real-world safety incident. "
            "Rejected predictions are excluded from confirmed findings."
        )
    }

    save_json(output_file, report)
    return report


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def containment_ratio(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    x1 = max(ix1, ox1)
    y1 = max(iy1, oy1)
    x2 = min(ix2, ox2)
    y2 = min(iy2, oy2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = box_area(inner_box)

    if area == 0:
        return 0.0

    return intersection / area


def associate_worker_findings(run_folder, adjudicated):
    detections_file = run_folder / "detections.json"
    adjudicated_file = run_folder / "adjudicated_findings.json"
    output_file = run_folder / "worker_level_findings.json"

    detection_report = load_json(detections_file)
    detections = detection_report.get("detections", [])

    people = [
        (index, detection)
        for index, detection in enumerate(detections)
        if detection.get("class_name") == "Person"
        and detection.get("bbox_xyxy")
    ]

    grouped = {}
    unassigned = []

    for finding in adjudicated.get("confirmed_findings", []):
        class_name = finding.get("class_name")

        if class_name not in ("no_vest", "no_gloves"):
            unassigned.append({
                "finding": finding,
                "reason": "No worker-association rule for this finding type."
            })
            continue

        source = None

        for detection in detections:
            if (
                detection.get("detector") == finding.get("detector")
                and detection.get("class_name") == class_name
                and abs(
                    detection.get("confidence", 0)
                    - finding.get("confidence", 0)
                ) < 0.00001
            ):
                source = detection
                break

        if source is None or not source.get("bbox_xyxy"):
            unassigned.append({
                "finding": finding,
                "reason": "Original detection or bounding box not found."
            })
            continue

        ppe_box = source["bbox_xyxy"]
        best_person = None
        best_ratio = 0.0

        for person_index, person in people:
            ratio = containment_ratio(
                ppe_box,
                person["bbox_xyxy"]
            )

            if ratio > best_ratio:
                best_ratio = ratio
                best_person = (person_index, person)

        if (
            best_person is None
            or best_ratio < CONTAINMENT_THRESHOLD
        ):
            unassigned.append({
                "finding": finding,
                "source_detection": source,
                "best_containment_ratio": best_ratio,
                "reason": "No sufficiently supported person association."
            })
            continue

        person_index, person = best_person
        group_key = (person_index, class_name)

        if group_key not in grouped:
            grouped[group_key] = {
                "worker_detection_index": person_index,
                "worker_confidence": person["confidence"],
                "worker_bbox_xyxy": person["bbox_xyxy"],
                "ppe_category": class_name,
                "human_decision": "Confirmed",
                "original_predictions": []
            }

        grouped[group_key]["original_predictions"].append({
            "candidate_index": finding.get("candidate_index"),
            "confidence": finding.get("confidence"),
            "detector": finding.get("detector"),
            "bbox_xyxy": source["bbox_xyxy"],
            "containment_ratio": round(best_ratio, 5),
            "reviewer_notes": finding.get("reviewer_notes", "")
        })

    worker_findings = []

    for group in grouped.values():
        predictions = group["original_predictions"]

        worker_findings.append({
            **group,
            "prediction_count": len(predictions),
            "confidence_values": [
                item["confidence"] for item in predictions
            ],
            "recommendation_eligible": True
        })

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_detection_report": str(detections_file),
        "source_adjudicated_report": str(adjudicated_file),
        "association_method": (
            "PPE bounding-box containment within detected person boxes."
        ),
        "containment_threshold": CONTAINMENT_THRESHOLD,
        "person_count": len(people),
        "confirmed_prediction_count": len(
            adjudicated.get("confirmed_findings", [])
        ),
        "worker_level_finding_count": len(worker_findings),
        "worker_level_findings": worker_findings,
        "unassigned_findings": unassigned,
        "interpretation_note": (
            "Associations are based on image bounding boxes, not verified "
            "worker identities. Original predictions and confidence scores "
            "are retained."
        )
    }

    save_json(output_file, report)
    return report


def build_safety_reasoning(run_folder, worker_report, adjudicated):
    output_file = run_folder / "safety_reasoning.json"

    worker_findings = worker_report.get(
        "worker_level_findings", []
    )

    recommendations = []

    for finding in worker_findings:
        class_name = finding.get("ppe_category")
        rule = RECOMMENDATION_RULES.get(class_name)

        recommendation = {
            "worker_detection_index": finding.get(
                "worker_detection_index"
            ),
            "worker_confidence": finding.get("worker_confidence"),
            "worker_bbox_xyxy": finding.get("worker_bbox_xyxy"),
            "ppe_category": class_name,
            "human_decision": finding.get("human_decision"),
            "prediction_count": finding.get("prediction_count"),
            "supporting_predictions": finding.get(
                "original_predictions", []
            )
        }

        if rule is None:
            recommendation.update({
                "recommendation_status": "requires_rule_review",
                "observation": (
                    "A worker-level finding was confirmed, but no "
                    "recommendation rule is defined for this category."
                ),
                "recommendation": None,
                "limitation": None
            })
        else:
            recommendation.update({
                "recommendation_status": "recommendation_generated",
                "observation": rule["observation"],
                "recommendation": rule["recommendation"],
                "limitation": rule["limitation"]
            })

        recommendations.append(recommendation)

    rejected = adjudicated.get("rejected_predictions", [])
    uncertain = adjudicated.get("uncertain_findings", [])
    pending = adjudicated.get("pending_findings", [])

    generated_count = sum(
        1 for item in recommendations
        if item["recommendation_status"] == "recommendation_generated"
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_worker_findings_report": str(
            run_folder / "worker_level_findings.json"
        ),
        "source_adjudicated_report": str(
            run_folder / "adjudicated_findings.json"
        ),
        "review_summary": adjudicated.get("summary", {}),
        "worker_level_summary": {
            "detected_people": worker_report.get("person_count", 0),
            "confirmed_predictions": worker_report.get(
                "confirmed_prediction_count", 0
            ),
            "worker_level_findings": len(worker_findings),
            "unassigned_findings": len(
                worker_report.get("unassigned_findings", [])
            )
        },
        "recommendations_generated": generated_count,
        "safety_recommendations": recommendations,
        "rejected_predictions": rejected,
        "uncertain_findings": uncertain,
        "pending_findings": pending,
        "interpretation_note": (
            "Recommendations are generated from human-confirmed "
            "worker-level findings. They are screening guidance, not "
            "formal safety certification, regulatory determinations, "
            "or proof of an unsafe condition. Site policies, work "
            "activities, and applicable requirements must be verified."
        )
    }

    save_json(output_file, report)
    return report


def process_reviewed_run(run_folder):
    adjudicated = build_adjudicated_report(run_folder)
    worker_report = associate_worker_findings(run_folder, adjudicated)
    reasoning = build_safety_reasoning(
        run_folder,
        worker_report,
        adjudicated
    )

    return adjudicated, worker_report, reasoning


# ---------------- STREAMLIT APP ----------------

st.set_page_config(
    page_title="Construction Safety AI",
    page_icon="🏗️",
    layout="wide"
)

st.title("Construction Safety AI")
st.subheader("Multi-Model Construction Site Screening")

st.info(
    "Detector outputs are predictions, not confirmed safety violations. "
    "Human review is required. Image-space relationships do not establish "
    "physical distance or prove a hazard."
)

st.header("1. Upload a construction-site image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

conf = st.slider(
    "Detection confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05
)

imgsz = st.select_slider(
    "Inference image size",
    options=[320, 416, 512, 640],
    value=640
)

if "run_folder" not in st.session_state:
    st.session_state.run_folder = None

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded construction-site image",
        width="stretch",
        alt="Image selected for construction safety screening"
    )

    if st.button("Run all three detectors", type="primary"):
        if not INFERENCE_SCRIPT.exists():
            st.error("unified_inference.py was not found.")
            st.stop()

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        folders_before = {
            folder.name
            for folder in RUNS_DIR.iterdir()
            if folder.is_dir()
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = Path(uploaded_file.name).suffix.lower()
        image_path = UPLOAD_DIR / f"uploaded_{timestamp}{suffix}"

        image_path.write_bytes(uploaded_file.getvalue())

        command = [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--image",
            str(image_path),
            "--conf",
            str(conf),
            "--imgsz",
            str(imgsz)
        ]

        with st.spinner(
            "Running PPE, Toolset, and Vehicle detectors on CPU..."
        ):
            try:
                result = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    check=False
                )

                if result.stdout:
                    st.code(result.stdout)

                if result.returncode != 0:
                    st.error("Inference failed.")
                    if result.stderr:
                        st.code(result.stderr)
                    st.stop()

                new_folders = [
                    folder
                    for folder in RUNS_DIR.iterdir()
                    if folder.is_dir()
                    and folder.name not in folders_before
                ]

                if not new_folders:
                    st.error(
                        "Inference completed, but no new output folder "
                        "was found."
                    )
                    st.stop()

                latest_folder = max(
                    new_folders,
                    key=lambda folder: folder.stat().st_mtime
                )

                st.session_state.run_folder = str(latest_folder)
                st.success("Inference completed successfully.")

            except Exception as error:
                st.error(f"Could not run inference: {error}")


run_folder_value = st.session_state.run_folder

if run_folder_value:
    run_folder = Path(run_folder_value)

    report_file = run_folder / "detections.json"
    annotated_file = run_folder / "unified_detections.jpg"
    review_file = run_folder / "human_review.json"
    adjudicated_file = run_folder / "adjudicated_findings.json"
    worker_file = run_folder / "worker_level_findings.json"
    reasoning_file = run_folder / "safety_reasoning.json"

    st.divider()
    st.header("2. Combined detector results")

    if not report_file.exists():
        st.error("The detection report is missing.")
        st.stop()

    report = load_json(report_file)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total detections",
        report.get("total_detections", 0)
    )

    col2.metric(
        "Review candidates",
        report.get("review_analysis", {}).get(
            "candidate_count", 0
        )
    )

    col3.metric(
        "Cross-model overlaps",
        len(report.get("cross_model_comparisons", []))
    )

    if annotated_file.exists():
        st.image(
            str(annotated_file),
            caption="Combined detections from all three models",
            width="stretch",
            alt="Combined detector predictions on the uploaded image"
        )

    with st.expander("Scene inventory"):
        inventory = report.get("scene_inventory", {})
        st.json(inventory.get("category_counts", {}))

    with st.expander("Person-machinery spatial analysis"):
        st.json(report.get("spatial_analysis", {}))

    with st.expander("Cross-model comparisons"):
        st.json(report.get("cross_model_comparisons", []))

    st.divider()
    st.header("3. Human review")

    candidates = report.get(
        "review_analysis", {}
    ).get("candidates", [])

    existing_decisions = {}

    if review_file.exists():
        saved = load_json(review_file)

        for item in saved.get("review_decisions", []):
            existing_decisions[item["candidate_index"]] = item

    st.write(f"Candidates requiring review: **{len(candidates)}**")

    if not candidates:
        st.success("No candidates require human review.")

    else:
        review_data = {
            "source_report": str(report_file),
            "review_decisions": []
        }

        with st.form("integrated_review_form"):
            for index, candidate in enumerate(candidates):
                st.divider()
                st.markdown(f"### Candidate {index + 1}")

                st.write("**Type:**", candidate.get("candidate_type"))
                st.write("**Status:**", candidate.get("status"))
                st.write("**Evidence:**", candidate.get("evidence"))
                st.write(
                    "**Interpretation:**",
                    candidate.get("interpretation")
                )

                if candidate.get("class_name"):
                    st.write(
                        "**Predicted class:**",
                        candidate["class_name"]
                    )

                if candidate.get("confidence") is not None:
                    st.write(
                        "**Confidence:**",
                        candidate["confidence"]
                    )

                previous = existing_decisions.get(index, {})

                options = [
                    "Pending",
                    "Confirmed",
                    "Rejected",
                    "Uncertain"
                ]

                previous_decision = previous.get(
                    "decision", "Pending"
                )

                if previous_decision not in options:
                    previous_decision = "Pending"

                decision = st.selectbox(
                    "Human review decision",
                    options,
                    index=options.index(previous_decision),
                    key=f"decision_{index}"
                )

                note = st.text_area(
                    "Reviewer notes",
                    value=previous.get("reviewer_notes", ""),
                    key=f"note_{index}"
                )

                review_data["review_decisions"].append({
                    "candidate_index": index,
                    "candidate": candidate,
                    "decision": decision,
                    "reviewer_notes": note
                })

            submitted = st.form_submit_button(
                "Save human review decisions",
                type="primary"
            )

        if submitted:
            save_json(review_file, review_data)
            st.success("Human review decisions saved.")
            st.code(str(review_file))

    # ---------------- FINAL REASONING PIPELINE ----------------

    st.divider()
    st.header("4. Generate worker-level reasoning")

    if not review_file.exists():
        st.warning(
            "Save your human review decisions before generating "
            "worker-level recommendations."
        )

    else:
        if st.button(
            "Generate adjudicated findings and recommendations",
            type="primary"
        ):
            try:
                with st.spinner(
                    "Processing human decisions, worker associations, "
                    "and safety recommendations..."
                ):
                    adjudicated, worker_report, reasoning = (
                        process_reviewed_run(run_folder)
                    )

                st.session_state.pipeline_completed = str(run_folder)
                st.success("All reasoning reports generated successfully.")

            except Exception as error:
                st.error(f"Reasoning pipeline failed: {error}")
                st.stop()

        if (
            st.session_state.get("pipeline_completed")
            == str(run_folder)
            and reasoning_file.exists()
        ):
            adjudicated = load_json(adjudicated_file)
            worker_report = load_json(worker_file)
            reasoning = load_json(reasoning_file)

            st.subheader("Adjudication summary")

            summary = adjudicated.get("summary", {})
            cols = st.columns(4)

            cols[0].metric(
                "Confirmed",
                summary.get("confirmed_predictions", 0)
            )
            cols[1].metric(
                "Rejected",
                summary.get("rejected_predictions", 0)
            )
            cols[2].metric(
                "Uncertain",
                summary.get("uncertain_predictions", 0)
            )
            cols[3].metric(
                "Pending",
                summary.get("pending_predictions", 0)
            )

            st.subheader("Worker-level findings")

            worker_summary = reasoning.get(
                "worker_level_summary", {}
            )

            cols = st.columns(4)
            cols[0].metric(
                "Detected people",
                worker_summary.get("detected_people", 0)
            )
            cols[1].metric(
                "Confirmed predictions",
                worker_summary.get("confirmed_predictions", 0)
            )
            cols[2].metric(
                "Worker-level findings",
                worker_summary.get("worker_level_findings", 0)
            )
            cols[3].metric(
                "Unassigned findings",
                worker_summary.get("unassigned_findings", 0)
            )

            for index, finding in enumerate(
                worker_report.get("worker_level_findings", [])
            ):
                with st.expander(
                    f"Worker detection #{finding.get('worker_detection_index')} "
                    f"— {finding.get('ppe_category')}"
                ):
                    st.write(
                        "**Supporting predictions:**",
                        finding.get("prediction_count")
                    )
                    st.write(
                        "**Association method:**",
                        "Bounding-box containment"
                    )
                    st.json(finding.get("original_predictions", []))

            st.subheader("Safety recommendations")

            recommendations = reasoning.get(
                "safety_recommendations", []
            )

            if not recommendations:
                st.info("No worker-level recommendations were generated.")

            for item in recommendations:
                st.markdown(
                    f"### {item.get('ppe_category')} "
                    f"— Worker #{item.get('worker_detection_index')}"
                )
                st.write("**Observation:**", item.get("observation"))
                st.write(
                    "**Recommendation:**",
                    item.get("recommendation")
                )
                st.caption(
                    f"Limitation: {item.get('limitation')}"
                )
                st.write(
                    "Supporting prediction count:",
                    item.get("prediction_count")
                )
                st.divider()

            with st.expander("Rejected predictions"):
                st.json(reasoning.get("rejected_predictions", []))

            with st.expander("Uncertain and pending findings"):
                st.write("Uncertain")
                st.json(reasoning.get("uncertain_findings", []))
                st.write("Pending")
                st.json(reasoning.get("pending_findings", []))

            st.subheader("Generated report files")

            for path in (
                adjudicated_file,
                worker_file,
                reasoning_file
            ):
                st.write(f"**{path.name}**")
                st.code(str(path))

            st.download_button(
                "Download safety reasoning JSON",
                data=reasoning_file.read_bytes(),
                file_name="safety_reasoning.json",
                mime="application/json"
            )

    st.divider()

    st.caption(
        "Original detector predictions and dataset annotations are not "
        "modified. Human decisions and derived reports are saved separately. "
        "Recommendations are screening guidance, not formal safety "
        "certification or regulatory determinations."
    )

    st.write("**Current inference run:**")
    st.code(str(run_folder))
