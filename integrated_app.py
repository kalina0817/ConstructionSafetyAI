
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw
import streamlit as st

from vision_language_reasoning import (
    MODELS,
    run_vision_language_analysis,
)

from screening_assessment import assess_worker_ppe, visual_safety_assessment

RULES = "Scene-graph rules"
DETECTION_MODELS = {"YOLO11n": "yolo", "RT-DETR-L": "rtdetr"}
VLM_FILES = {"MiniCPM-V": "vision_language_minicpm.json", "Qwen3-VL": "vision_language_qwen.json"}

ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = ROOT / "unified_inference.py"
RUNS_DIR = ROOT / "reports" / "inference_runs"


st.set_page_config(
    page_title="Construction Safety AI",
    page_icon=":material/engineering:",
    layout="wide",
)

st.title("Check a site photo")
st.write("Choose how the app finds objects and how it checks for possible hazards.")


def display_cross_model_disagreement(candidate, report, annotated_file):
    """Display the original bounding boxes for two disagreeing detectors."""

    detections = report.get("detections", [])

    index_a = candidate.get("source_detection_index_a")
    index_b = candidate.get("source_detection_index_b")

    if not isinstance(index_a, int) or not isinstance(index_b, int):
        st.warning("The original detection locations are unavailable.")
        return

    if not (
        0 <= index_a < len(detections)
        and 0 <= index_b < len(detections)
    ):
        st.warning("The original model predictions could not be retrieved.")
        return

    detection_a = detections[index_a]
    detection_b = detections[index_b]

    col_a, col_b = st.columns(2)

    with col_a:
        st.write(f"**{candidate.get('detector_a', 'Detector A')}**")
        st.write(f"Class: {candidate.get('class_a', 'Unknown')}")
        st.write(f"Model confidence: {candidate.get('confidence_a')}")
        st.write(f"Bounding box: {detection_a.get('box_xyxy')}")

    with col_b:
        st.write(f"**{candidate.get('detector_b', 'Detector B')}**")
        st.write(f"Class: {candidate.get('class_b', 'Unknown')}")
        st.write(f"Model confidence: {candidate.get('confidence_b')}")
        st.write(f"Bounding box: {detection_b.get('box_xyxy')}")

    iou = candidate.get("iou")

    if iou is not None:
        st.metric("Overlap between model boxes", f"{iou:.4f}")

    if not annotated_file.is_file():
        st.warning("The image with detection boxes is unavailable.")
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
            width="stretch",
        )

    except (OSError, ValueError, TypeError) as error:
        st.warning(f"Could not display the comparison image: {error}")


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

    st.subheader("Overall result")

    if outcome == "POTENTIAL":
        st.warning(f"Possible safety concerns found: {finding_count}")

    elif outcome == "HIGH_CONFIDENCE_ALERT":
        st.error(f"Strong evidence of a possible safety concern: {finding_count}")

    elif outcome == "NO_SUPPORTED":
        st.success("The app found no concern it could support")

    elif outcome == "INSUFFICIENT":
        st.info("Not enough model information to complete the checks")

    else:
        st.info("The overall result is unavailable")

    st.write(explanation)

    if outcome == "NO_SUPPORTED":
        st.caption("This does not mean the construction site is safe.")

    elif outcome == "INSUFFICIENT":
        st.caption("Some checks could not be completed with the selected models.")

    high_confidence_available = screening_outcome.get(
        "high_confidence_alert_available",
        False,
    )

    if not high_confidence_available:
        limitation = screening_outcome.get(
            "high_confidence_alert_limitation"
        )

        if limitation:
            st.caption(f"Why a strong alert was not available: {limitation}")


photo, settings = st.columns([5, 4], gap="large")
with photo, st.container(border=True):
    st.subheader("Upload a site photo")
    uploaded_file = st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, width="stretch", alt="Photo selected for screening")

with settings, st.container(border=True):
    st.subheader("Choose your models")
    object_detection_model = st.selectbox(
        "Detection model",
        list(DETECTION_MODELS),
        key="object_detection_model",
        help="Finds workers, protective gear, tools, and vehicles in the photo.",
    )

    selected_reasoning = st.selectbox(
        "Reasoning model",
        [RULES, *MODELS],
        key="reasoning_model",
        help="Checks what the objects mean together. Scene-graph rules use their positions. MiniCPM-V and Qwen3-VL also look at the photo.",
    )
    with st.expander("More photo settings"):
        conf = st.slider(
            "Minimum score to keep an object",
            min_value=0.05,
            max_value=0.90,
            value=0.25,
            step=0.05,
        )

        imgsz = st.select_slider(
            "Photo size for the check (pixels)",
            options=[320, 416, 512, 640],
            value=640,
        )
    run_requested = st.button("Run Screening", type="primary", icon=":material/image_search:",
                              width="stretch", disabled=uploaded_file is None)
    st.caption("Confirm potential hazards on site. A photo alone cannot establish physical distance.")

if "run_folder" not in st.session_state:
    st.session_state.run_folder = None

if "image_path" not in st.session_state:
    st.session_state.image_path = None


if run_requested and uploaded_file is not None:
    st.session_state.run_folder = None
    if not INFERENCE_SCRIPT.is_file():
        st.error("The image analysis script could not be found.")
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
        "--image", str(image_path),
        "--conf", str(conf),
        "--imgsz", str(imgsz),
        "--output-dir", str(output_dir),
        "--screening-model", DETECTION_MODELS[object_detection_model],
    ]

    if selected_reasoning != RULES:
        command.append("--graph-only")

    with st.spinner("Detecting objects and building scene evidence..."):
        try:
            result = subprocess.run(
                command,
                cwd=str(ROOT),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                check=False,
            )

            if result.stdout:
                st.code(result.stdout)

            if result.returncode != 0:
                st.error("The selected detector could not complete screening.")
                if result.stderr:
                    st.code(result.stderr)
                st.stop()

            report_file = output_dir / "detections.json"
            if not report_file.is_file():
                st.error(
                    "Detection finished, but detections.json was not created."
                )
                st.stop()

            with report_file.open("r", encoding="utf-8") as file:
                report_for_reasoning = json.load(file)

        except Exception as error:
            st.error(f"Could not complete object detection: {error}")
            st.stop()

    assessment = {
        "detection_model": object_detection_model,
        "detection_checkpoints": report_for_reasoning["screening_checkpoints"],
        "reasoning_model": selected_reasoning,
        "reasoning_model_id": MODELS.get(selected_reasoning, "scene_graph_rules"),
        "reasoning_status": "COMPLETED",
        "reasoning_error": None,
    }
    graph = report_for_reasoning["scene_graph"]
    if selected_reasoning != RULES:
        with st.spinner(f"Running {selected_reasoning} visual reasoning..."):
            try:
                vlm_result = run_vision_language_analysis(
                    image_path=image_path,
                    report=report_for_reasoning,
                    model_label=selected_reasoning,
                    output_file=output_dir / VLM_FILES[selected_reasoning],
                    screening=True,
                )
                assessment["reasoning_status"] = vlm_result["status"]
                assessment["reasoning_error"] = vlm_result["error"]
                if vlm_result["status"] == "COMPLETED":
                    assessment["visual_reasoning"] = vlm_result["analysis"]
            except Exception as error:
                assessment["reasoning_status"] = "ERROR"
                assessment["reasoning_error"] = str(error)

    assessment["worker_ppe"] = assess_worker_ppe(
        graph,
        (assessment.get("visual_reasoning", {}).get("worker_ppe") or [])
        if selected_reasoning != RULES else None,
    )
    if selected_reasoning != RULES:
        assessment.update(visual_safety_assessment(graph, assessment))
    (output_dir / "screening_report.json").write_text(
        json.dumps({**report_for_reasoning, **assessment}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    st.session_state.run_folder = str(output_dir)
    st.session_state.image_path = str(image_path)

run_folder_value = st.session_state.run_folder


if run_folder_value:
    run_folder = Path(run_folder_value)
    annotated_file = run_folder / "unified_detections.jpg"
    review_file = run_folder / "human_review.json"

    st.divider()
    st.header("Screening report")

    export_file = run_folder / "screening_report.json"
    try:
        screening = json.loads(export_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read the screening report. Run Screening again. {error}")
        st.stop()
    model_info, export = st.columns([3, 2], vertical_alignment="center")
    model_info.write(f"**Detection model** · {screening['detection_model']}")
    model_info.write(f"**Reasoning model** · {screening['reasoning_model']}")
    if screening["reasoning_error"]:
        st.error(f"{screening['reasoning_model']} did not complete. {screening['reasoning_error']}")
    export.download_button(
        "Download screening report",
        data=export_file.read_bytes(),
        file_name=f"screening_{run_folder.name}.json",
        mime="application/json",
    )
    report = screening
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

    display_screening_outcome(screening_outcome, findings)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Objects found",
        report.get("total_detections", 0),
        border=True,
    )

    col2.metric(
        "Potential safety alerts",
        len(findings),
        border=True,
    )

    col3.metric(
        "Overlapping labels",
        len(report.get("cross_model_comparisons", [])),
        border=True,
    )

    findings_tab, evidence_tab, details_tab, review_tab = st.tabs(
        ["Safety findings", "Photo and workers", "Full records", "Your review"]
    )
    with evidence_tab:
        if annotated_file.is_file():
            st.image(
                str(annotated_file),
                caption="Objects found by the selected AI models",
                width="stretch",
                alt="Combined detector predictions on the uploaded image",
            )

        st.subheader("Protective gear by worker")
        if screening["worker_ppe"]:
            gear_rows = []
            for item in screening["worker_ppe"]:
                status = item.get("status")
                gear_rows.append({
                    "Worker in photo": item.get("worker"),
                    "Protective gear": str(item.get("item", "")).replace("_", " "),
                    "What the photo shows": {"unassessable": "Cannot tell", "present": "Visible",
                                             "potentially_missing": "May be missing"}.get(status, status),
                    "What the app saw": item.get("visual_basis_or_uncertainty"),
                    "What to check": item.get("preventive_recommendation"),
                })
            st.dataframe(gear_rows, hide_index=True)
        else:
            st.info("The app could not identify a worker to check.")


        if "machinery_coverage" in screening:
            st.subheader("Workers near machines")
            observations = screening.get("visual_reasoning", {}).get("worker_context", [])
            if observations:
                st.caption("The app looked at each worker up close and in the full photo. Some machines mentioned here may have no box in the photo.")
                st.dataframe(observations, hide_index=True)
            coverage = screening["machinery_coverage"]
            st.caption(f"Checked {coverage['assessed_pairs']} of {coverage['candidate_pairs']} possible worker and machine pairs. "
                       + coverage["limitation"])
            if screening["worker_machinery_assessments"]:
                st.dataframe(screening["worker_machinery_assessments"], hide_index=True)

    with findings_tab:
        st.subheader("Potential hazards")

        if screening["reasoning_status"] != "COMPLETED":
            st.info("The selected reasoning model did not produce a safety assessment.")
        elif not findings and screening_outcome.get("outcome") == "INSUFFICIENT":
            st.info("The available evidence was insufficient to resolve all checks.")
        elif not findings:
            st.info(
                "The AI did not find any concerns covered by its current checks."
            )

            st.caption(
                "This does not mean the construction site is safe."
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
                    label = {"potential_missing_ppe": "Protective gear may be missing",
                             "potential_worker_machinery_interaction": "Worker near machinery"}.get(
                                 category, str(category).replace("_", " ").capitalize())
                    st.subheader(f"{index}. {label}")

                    st.write(
                        "**Status:**",
                        str(status).replace("_", " ").title(),
                    )

                    if finding.get("review_required"):
                        st.caption("This concern needs a closer look on site.")

                    if finding.get("evidence"):
                        st.write("**What the app found**", finding["evidence"])

                    if finding.get("observation"):
                        st.write(
                            "**What this may mean**",
                            finding["observation"],
                        )

                    if finding.get("recommendation"):
                        st.write(
                            "**What to do next**",
                            finding["recommendation"],
                        )

                    if finding.get("limitation"):
                        st.write(
                            "**What the photo cannot show**",
                            finding["limitation"],
                        )

                    uncertainty = finding.get("uncertainty_reasons", [])

                    if uncertainty:
                        st.write("**What is uncertain:**")

                        if isinstance(uncertainty, list):
                            for reason in uncertainty:
                                st.write(f"- {reason}")
                        else:
                            st.write(uncertainty)

                    if finding.get("interpretation"):
                        st.write(
                            "**Possible meaning:**",
                            finding["interpretation"],
                        )

    with details_tab:
        st.subheader("How the app reached its answer")

        with st.expander("Objects found in the picture"):
            inventory = report.get("scene_inventory", {})

            if not isinstance(inventory, dict):
                inventory = {}

            st.json(inventory.get("category_counts", {}))

        with st.expander("Person–machinery spatial analysis"):
            st.json(report.get("spatial_analysis", {}))

        with st.expander("How detected objects are related"):
            st.json(report.get("scene_graph", {}))

        with st.expander("Full safety check details"):
            st.json(safety_assessment)

        with st.expander("Where the AI models disagree or overlap"):
            st.json(report.get("cross_model_comparisons", []))

        with st.expander("Details about the overall result"):
            st.json(screening_outcome)

    with review_tab:
        st.subheader("Review findings")

        st.write(
            "You can review the AI's predictions and record whether you "
            "agree with them. This step is optional. The automatic alerts "
            "are already available without this review."
        )

        if not review_candidates:
            st.caption(
                "There are no items available for optional review in this image."
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
                f"Review {len(review_candidates)} items (optional)"
            ):
                review_data = {
                    "source_report": str((run_folder / "detections.json").resolve()),
                    "review_decisions": [],
                }

                with st.form("optional_review_form"):
                    for index, candidate in enumerate(review_candidates):
                        st.divider()
                        st.markdown(f"#### Item {index + 1}")

                        if candidate.get("candidate_type") == (
                            "cross_model_class_disagreement_review"
                        ):
                            st.warning(
                                "The two AI models gave different labels"
                            )

                            display_cross_model_disagreement(
                                candidate,
                                report,
                                annotated_file,
                            )

                            st.caption(
                                "The models may be looking at the same "
                                "object but giving it different labels. "
                                "Neither model's label is automatically correct."
                            )

                        st.write(
                            "**Type of item:**",
                            candidate.get("candidate_type", "Not specified"),
                        )

                        st.write(
                            "**What the app found**",
                            candidate.get("evidence", "Not specified"),
                        )

                        st.write(
                            "**Possible meaning:**",
                            candidate.get("interpretation", "Not specified"),
                        )

                        if candidate.get("class_name"):
                            st.write(
                                "**AI label:**",
                                candidate["class_name"],
                            )

                        if candidate.get("confidence") is not None:
                            st.write(
                                "**Model confidence:**",
                                candidate["confidence"],
                            )

                        if candidate.get("person_class"):
                            st.write(
                                "**Worker detected:**",
                                candidate["person_class"],
                                candidate.get("person_detector", ""),
                                candidate.get("person_confidence", ""),
                            )

                        if candidate.get("machinery_class"):
                            st.write(
                                "**Machinery detected:**",
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
                            "Your review decision",
                            options,
                            index=options.index(previous_decision),
                            key=f"optional_decision_{index}",
                        )

                        note = st.text_area(
                            "Your review notes (optional)",
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
                        "Save review decisions",
                    )

                if save_review:
                    try:
                        with review_file.open(
                            "w",
                            encoding="utf-8",
                        ) as file:
                            json.dump(review_data, file, indent=4)

                        st.success("Your review decisions were saved.")

                    except OSError as error:
                        st.error(f"Could not save your review: {error}")

        st.divider()

        st.caption(
            "Alerts are created before optional human review. AI predictions "
            "do not prove that PPE is missing, that machinery is operating, "
            "or that a safety violation occurred. Image positions do not show "
            "actual physical distance."
        )

        st.write("**Current inference run:**")
        st.code(str(run_folder))

        if review_file.is_file():
            st.caption(
                "An optional verification file exists for this inference run."
            )

        st.divider()
        st.subheader("Follow-up report")

        st.write(
            "This optional step uses your saved review decisions to prepare "
            "a follow-up report and recommendations. It is not needed for "
            "the AI to create or display alerts."
        )

        if not review_file.is_file():
            st.info(
                "Save your optional review decisions first if you want "
                "to create the follow-up report."
            )

        elif st.button(
            "Create follow-up report and recommendations",
        ):
            pipeline_scripts = [
                "build_adjudicated_report.py",
                "associate_worker_findings.py",
                "safety_reasoning.py",
            ]

            pipeline_success = True

            for script_name in pipeline_scripts:
                script_path = ROOT / script_name

                st.write(f"Running follow-up step: {script_name}")

                if not script_path.is_file():
                    st.error(f"Required follow-up script was not found: {script_path}")
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
                    st.error(f"The follow-up step failed: {script_name}")

                    if result.stderr:
                        st.code(result.stderr)

                    pipeline_success = False
                    break

            if pipeline_success:
                st.success(
                    "The follow-up review and recommendation steps completed."
                )
