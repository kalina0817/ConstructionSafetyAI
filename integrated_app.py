
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw
import streamlit as st


ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = ROOT / "unified_inference.py"
RUNS_DIR = ROOT / "reports" / "inference_runs"


st.set_page_config(
    page_title="Construction Safety AI",
    page_icon="🏗️",
    layout="wide",
)

st.title("Construction Safety AI")
st.subheader("Multi-Model Construction Site Screening")

st.info(
    "The system generates potential safety alerts automatically. "
    "Human verification is optional. Detector predictions and "
    "image-space relationships do not establish confirmed safety "
    "violations or actual physical distance."
)


def display_cross_model_disagreement(candidate, report, annotated_file):
    """Display the original bounding boxes for two disagreeing detectors."""

    detections = report.get("detections", [])

    index_a = candidate.get("source_detection_index_a")
    index_b = candidate.get("source_detection_index_b")

    if not isinstance(index_a, int) or not isinstance(index_b, int):
        st.warning("Original detection indices are unavailable.")
        return

    if not (
        0 <= index_a < len(detections)
        and 0 <= index_b < len(detections)
    ):
        st.warning("Could not retrieve the original detector predictions.")
        return

    detection_a = detections[index_a]
    detection_b = detections[index_b]

    col_a, col_b = st.columns(2)

    with col_a:
        st.write(f"**{candidate.get('detector_a', 'Detector A')}**")
        st.write(f"Class: {candidate.get('class_a', 'Unknown')}")
        st.write(f"Confidence: {candidate.get('confidence_a')}")
        st.write(f"Bounding box: {detection_a.get('box_xyxy')}")

    with col_b:
        st.write(f"**{candidate.get('detector_b', 'Detector B')}**")
        st.write(f"Class: {candidate.get('class_b', 'Unknown')}")
        st.write(f"Confidence: {candidate.get('confidence_b')}")
        st.write(f"Bounding box: {detection_b.get('box_xyxy')}")

    iou = candidate.get("iou")

    if iou is not None:
        st.metric("Bounding-box IoU", f"{iou:.4f}")

    if not annotated_file.is_file():
        st.warning("The annotated image is unavailable.")
        return

    try:
        image = Image.open(annotated_file).convert("RGB")
        draw = ImageDraw.Draw(image)

        comparisons = [
            (detection_a, "red", candidate.get("class_a", "Detector A")),
            (detection_b, "blue", candidate.get("class_b", "Detector B")),
        ]

        for detection, color, label in comparisons:
            box = detection.get("box_xyxy")

            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            x1, y1, x2, y2 = map(int, box)

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color,
                width=5,
            )

            draw.text(
                (x1, max(0, y1 - 18)),
                label,
                fill=color,
            )

        st.image(
            image,
            caption=(
                "Cross-model disagreement: "
                "red = detector A; blue = detector B"
            ),
            use_container_width=True,
        )

    except (OSError, ValueError, TypeError) as error:
        st.warning(f"Could not display comparison image: {error}")


def display_screening_outcome(screening_outcome, findings):
    """Display the overall automatic screening classification."""

    if not isinstance(screening_outcome, dict):
        screening_outcome = {}

    outcome = screening_outcome.get("outcome", "UNKNOWN")
    explanation = screening_outcome.get(
        "explanation",
        "The screening outcome is unavailable in this report.",
    )
    finding_count = screening_outcome.get(
        "finding_count",
        len(findings),
    )

    st.subheader("Overall screening outcome")

    if outcome == "POTENTIAL":
        st.warning(
            f"Potential findings detected: {finding_count}"
        )

    elif outcome == "NO_SUPPORTED":
        st.success("No supported findings generated")

    elif outcome == "INSUFFICIENT":
        st.info("Insufficient detector coverage")

    else:
        st.info("Screening outcome unavailable")

    st.write(explanation)

    if outcome == "NO_SUPPORTED":
        st.caption(
            "No supported finding does not mean the image or "
            "construction site is safe."
        )

    elif outcome == "INSUFFICIENT":
        st.caption(
            "Some screening rules could not be evaluated with "
            "the selected detector configuration."
        )

    high_confidence_available = screening_outcome.get(
        "high_confidence_alert_available",
        False,
    )

    if not high_confidence_available:
        limitation = screening_outcome.get(
            "high_confidence_alert_limitation"
        )

        if limitation:
            st.caption(
                f"High-confidence alert limitation: {limitation}"
            )


st.header("1. Upload a construction-site image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

conf = st.slider(
    "Detection confidence threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05,
)

imgsz = st.select_slider(
    "Inference image size",
    options=[320, 416, 512, 640],
    value=640,
)

st.subheader("Select detector architecture")

col1, col2, col3 = st.columns(3)

with col1:
    ppe_model = st.selectbox(
        "PPE model",
        ["YOLO11n", "RT-DETR-L"],
        key="ppe_model",
    )

with col2:
    toolset_model = st.selectbox(
        "Toolset-35 model",
        ["YOLO11n", "RT-DETR-L"],
        key="toolset_model",
    )

with col3:
    vehicles_model = st.selectbox(
        "Vehicles model",
        ["YOLO11n", "RT-DETR-L"],
        key="vehicles_model",
    )

if "run_folder" not in st.session_state:
    st.session_state.run_folder = None


if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded construction-site image",
        width="stretch",
        alt="Image selected for construction safety screening",
    )

    if st.button("Run selected detectors", type="primary"):
        if not INFERENCE_SCRIPT.is_file():
            st.error("unified_inference.py was not found.")
            st.stop()

        upload_dir = ROOT / "input_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = Path(uploaded_file.name).suffix.lower()

        image_path = upload_dir / f"uploaded_{timestamp}{suffix}"
        output_dir = RUNS_DIR / timestamp

        image_path.write_bytes(uploaded_file.getvalue())

        command = [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--image",
            str(image_path),
            "--conf",
            str(conf),
            "--imgsz",
            str(imgsz),
            "--output-dir",
            str(output_dir),
            "--ppe-model",
            "yolo" if ppe_model == "YOLO11n" else "rtdetr",
            "--toolset-model",
            "yolo" if toolset_model == "YOLO11n" else "rtdetr",
            "--vehicles-model",
            "yolo" if vehicles_model == "YOLO11n" else "rtdetr",
        ]

        with st.spinner(
            "Running the selected PPE, Toolset-35, and Vehicle models on CPU..."
        ):
            try:
                result = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.stdout:
                    st.code(result.stdout)

                if result.returncode != 0:
                    st.error("Inference failed.")

                    if result.stderr:
                        st.code(result.stderr)

                    st.stop()

                if not (output_dir / "detections.json").is_file():
                    st.error(
                        "Inference completed without creating the "
                        "expected detection report."
                    )
                    st.stop()

                st.session_state.run_folder = str(output_dir)
                st.success("Inference completed successfully.")

            except Exception as error:
                st.error(f"Could not run inference: {error}")


run_folder_value = st.session_state.run_folder


if run_folder_value:
    run_folder = Path(run_folder_value)
    report_file = run_folder / "detections.json"
    annotated_file = run_folder / "unified_detections.jpg"
    review_file = run_folder / "human_review.json"

    st.divider()
    st.header("2. Combined detector results")

    if not report_file.is_file():
        st.error("The detection report is missing.")
        st.stop()

    try:
        with report_file.open("r", encoding="utf-8") as file:
            report = json.load(file)

    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read the detection report: {error}")
        st.stop()

    safety_assessment = report.get("safety_assessment", {})

    if not isinstance(safety_assessment, dict):
        safety_assessment = {}

    findings = safety_assessment.get("findings", [])

    if not isinstance(findings, list):
        findings = []

    screening_outcome = report.get("screening_outcome", {})

    if not isinstance(screening_outcome, dict):
        screening_outcome = {}

    review_analysis = report.get("review_analysis", {})

    if not isinstance(review_analysis, dict):
        review_analysis = {}

    review_candidates = review_analysis.get("candidates", [])

    if not isinstance(review_candidates, list):
        review_candidates = []

    st.subheader("Overall screening outcome")

    display_screening_outcome(screening_outcome, findings)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total detections",
        report.get("total_detections", 0),
    )

    col2.metric(
        "Potential safety alerts",
        len(findings),
    )

    col3.metric(
        "Cross-model overlaps",
        len(report.get("cross_model_comparisons", [])),
    )

    if annotated_file.is_file():
        st.image(
            str(annotated_file),
            caption="Combined detections from all three models",
            width="stretch",
            alt="Combined detector predictions on the uploaded image",
        )

    st.divider()
    st.header("3. Automatic safety screening results")

    st.write(
        "The following findings are generated automatically from "
        "the detector outputs and the available image-space evidence. "
        "They do not depend on human review."
    )

    if not findings:
        st.success(
            "No supported potential safety alerts were generated "
            "for this image."
        )

        st.caption(
            "This does not establish that the construction site is safe."
        )

    else:
        for index, finding in enumerate(findings, start=1):
            category = finding.get(
                "category",
                finding.get(
                    "finding_category",
                    "Potential safety concern",
                ),
            )

            status = finding.get("status", "potential")

            with st.container(border=True):
                st.subheader(f"Potential alert {index}: {category}")

                st.write(
                    "**Status:**",
                    str(status).replace("_", " ").title(),
                )

                st.write(
                    "**Alert generated automatically:**",
                    finding.get("alert_generated_automatically", True),
                )

                st.write(
                    "**Human review required:**",
                    finding.get("review_required", False),
                )

                if finding.get("rule_id"):
                    st.write("**Screening rule:**", finding["rule_id"])

                if finding.get("evidence"):
                    st.write("**Evidence:**", finding["evidence"])

                if finding.get("observation"):
                    st.write(
                        "**Observation:**",
                        finding["observation"],
                    )

                if finding.get("recommendation"):
                    st.write(
                        "**Recommended action:**",
                        finding["recommendation"],
                    )

                if finding.get("limitation"):
                    st.write(
                        "**Limitation:**",
                        finding["limitation"],
                    )

                uncertainty = finding.get("uncertainty_reasons", [])

                if uncertainty:
                    st.write("**Uncertainty:**")

                    if isinstance(uncertainty, list):
                        for reason in uncertainty:
                            st.write(f"- {reason}")
                    else:
                        st.write(uncertainty)

                if finding.get("interpretation"):
                    st.write(
                        "**Interpretation:**",
                        finding["interpretation"],
                    )

    st.divider()
    st.header("4. Scene analysis")

    with st.expander("Scene inventory"):
        inventory = report.get("scene_inventory", {})

        if not isinstance(inventory, dict):
            inventory = {}

        st.json(inventory.get("category_counts", {}))

    with st.expander("Person–machinery spatial analysis"):
        st.json(report.get("spatial_analysis", {}))

    with st.expander("Scene graph and image-space relationships"):
        st.json(report.get("scene_graph", {}))

    with st.expander("Complete safety assessment"):
        st.json(safety_assessment)

    with st.expander("Cross-model comparisons"):
        st.json(report.get("cross_model_comparisons", []))

    with st.expander("Screening outcome details"):
        st.json(screening_outcome)

    st.divider()
    st.header("5. Optional human verification")

    st.write(
        "You can independently verify detector predictions if needed. "
        "This step is optional and does not control whether automatic "
        "safety findings are displayed."
    )

    if not review_candidates:
        st.caption(
            "There are no optional verification candidates in this run."
        )

    else:
        existing_decisions = {}

        if review_file.is_file():
            try:
                with review_file.open("r", encoding="utf-8") as file:
                    saved_review = json.load(file)

                for item in saved_review.get("review_decisions", []):
                    existing_decisions[item["candidate_index"]] = item

            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ):
                existing_decisions = {}

        with st.expander(
            f"Optional verification of {len(review_candidates)} candidates"
        ):
            review_data = {
                "source_report": str(report_file.resolve()),
                "review_decisions": [],
            }

            with st.form("optional_review_form"):
                for index, candidate in enumerate(review_candidates):
                    st.divider()
                    st.markdown(f"#### Candidate {index + 1}")

                    if candidate.get("candidate_type") == (
                        "cross_model_class_disagreement_review"
                    ):
                        st.warning(
                            "Optional cross-model class disagreement"
                        )

                        display_cross_model_disagreement(
                            candidate,
                            report,
                            annotated_file,
                        )

                        st.caption(
                            "Overlapping predictions may refer to the "
                            "same physical object. Neither detector's "
                            "class is treated as ground truth."
                        )

                    st.write(
                        "**Type:**",
                        candidate.get("candidate_type", "Not specified"),
                    )

                    st.write(
                        "**Evidence:**",
                        candidate.get("evidence", "Not specified"),
                    )

                    st.write(
                        "**Interpretation:**",
                        candidate.get("interpretation", "Not specified"),
                    )

                    if candidate.get("class_name"):
                        st.write(
                            "**Predicted class:**",
                            candidate["class_name"],
                        )

                    if candidate.get("confidence") is not None:
                        st.write(
                            "**Confidence:**",
                            candidate["confidence"],
                        )

                    if candidate.get("person_class"):
                        st.write(
                            "**Person detection:**",
                            candidate["person_class"],
                            candidate.get("person_detector", ""),
                            candidate.get("person_confidence", ""),
                        )

                    if candidate.get("machinery_class"):
                        st.write(
                            "**Machinery detection:**",
                            candidate["machinery_class"],
                            candidate.get("machinery_detector", ""),
                            candidate.get("machinery_confidence", ""),
                        )

                    previous = existing_decisions.get(index, {})

                    options = [
                        "Pending",
                        "Confirmed",
                        "Rejected",
                        "Uncertain",
                    ]

                    previous_decision = previous.get(
                        "decision",
                        "Pending",
                    )

                    if previous_decision not in options:
                        previous_decision = "Pending"

                    decision = st.selectbox(
                        "Optional verification decision",
                        options,
                        index=options.index(previous_decision),
                        key=f"optional_decision_{index}",
                    )

                    note = st.text_area(
                        "Optional verification notes",
                        value=previous.get("reviewer_notes", ""),
                        key=f"optional_note_{index}",
                    )

                    review_data["review_decisions"].append({
                        "candidate_index": index,
                        "candidate": candidate,
                        "decision": decision,
                        "reviewer_notes": note,
                    })

                save_review = st.form_submit_button(
                    "Save optional verification",
                )

            if save_review:
                try:
                    with review_file.open(
                        "w",
                        encoding="utf-8",
                    ) as file:
                        json.dump(review_data, file, indent=4)

                    st.success("Optional verification decisions saved.")

                except OSError as error:
                    st.error(f"Could not save verification: {error}")

    st.divider()

    st.caption(
        "Automatic findings are generated before optional verification. "
        "Detector predictions, image-space proximity, and object "
        "relationships do not prove physical distance, PPE absence, "
        "machine operation, or a confirmed safety violation."
    )

    st.write("**Current inference run:**")
    st.code(str(run_folder))

    if review_file.is_file():
        st.caption(
            "An optional verification file exists for this inference run."
        )

    st.divider()
    st.header("6. Optional adjudication and recommendations")

    st.write(
        "This separate workflow uses saved verification decisions. "
        "It is not needed to generate or display automatic potential alerts."
    )

    if not review_file.is_file():
        st.info(
            "Save optional verification decisions first if you want "
            "to run the adjudication workflow."
        )

    elif st.button(
        "Run optional adjudication and recommendations",
    ):
        pipeline_scripts = [
            "build_adjudicated_report.py",
            "associate_worker_findings.py",
            "safety_reasoning.py",
        ]

        pipeline_success = True

        for script_name in pipeline_scripts:
            script_path = ROOT / script_name

            st.write(f"Running: {script_name}")

            if not script_path.is_file():
                st.error(f"Script not found: {script_path}")
                pipeline_success = False
                break

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(ROOT),
                env={
                    **os.environ,
                    "CONSTRUCTION_SAFETY_RUN_FOLDER": str(
                        run_folder.resolve()
                    ),
                },
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            if result.stdout:
                st.code(result.stdout)

            if result.returncode != 0:
                st.error(f"{script_name} failed.")

                if result.stderr:
                    st.code(result.stderr)

                pipeline_success = False
                break

        if pipeline_success:
            st.success(
                "Optional adjudication, worker association, and "
                "recommendation workflow completed."
            )
