
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = ROOT / "unified_inference.py"
RUNS_DIR = ROOT / "reports" / "inference_runs"

st.set_page_config(
    page_title="Construction Safety AI",
    page_icon="🏗️",
    layout="wide"
)

st.title("Construction Safety AI")
st.subheader("Multi-Model Construction Site Screening")

st.info(
    "Detector outputs are predictions, not confirmed safety violations. "
    "Review is required. Image-space relationships do not establish "
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

                run_folder = output_dir
                if not (run_folder / "detections.json").is_file():
                    st.error(
                        "Inference completed without creating the expected "
                        "detection report."
                    )
                    st.stop()

                st.session_state.run_folder = str(run_folder)

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

    if not report_file.exists():
        st.error("The detection report is missing.")
        st.stop()

    with report_file.open("r", encoding="utf-8") as file:
        report = json.load(file)
    # Automatically create a human review file for every new inference run.
    if not review_file.exists():
        candidates_for_review = report.get(
            "review_analysis", {}
        ).get("candidates", [])

        initial_review = {
            "source_report": str(report_file.resolve()),
            "review_decisions": [
                {
                    "candidate_index": index,
                    "candidate": candidate,
                    "decision": "Pending",
                    "reviewer_notes": ""
                }
                for index, candidate in enumerate(candidates_for_review)
            ]
        }

        with review_file.open("w", encoding="utf-8") as file:
            json.dump(initial_review, file, indent=4)

        st.success("Human review file created automatically.")

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

    with st.expander("Person–machinery spatial analysis"):
        st.json(report.get("spatial_analysis", {}))

    with st.expander("Scene graph and image-space relationships"):
        st.json(report.get("scene_graph", {}))

    with st.expander("Rule-based screening and uncertainty"):
        st.json(report.get("safety_assessment", {}))

    with st.expander("Cross-model comparisons"):
        st.json(report.get("cross_model_comparisons", []))

    st.divider()
    st.header("3. Human review")

    candidates = report.get(
        "review_analysis", {}
    ).get("candidates", [])

    existing_decisions = {}

    if review_file.exists():
        with review_file.open("r", encoding="utf-8") as file:
            saved = json.load(file)

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

                if candidate.get("person_class"):
                    st.write(
                        "**Person detection:**",
                        candidate["person_class"],
                        f"({candidate.get('person_detector')}, "
                        f"{candidate.get('person_confidence'):.3f})"
                    )
                    st.write(
                        "**Machinery detection:**",
                        candidate["machinery_class"],
                        f"({candidate.get('machinery_detector')}, "
                        f"{candidate.get('machinery_confidence'):.3f})"
                    )
                    st.write(
                        "**Normalized image-space distance:**",
                        candidate["normalized_image_distance"]
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
            with review_file.open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(review_data, file, indent=4)

            st.success("Human review decisions saved.")
            st.code(str(review_file))

    st.divider()

    st.caption(
        "Original detector predictions and dataset annotations are "
        "not modified. Human decisions are saved separately."
    )

    st.write("**Current inference run:**")
    st.code(str(run_folder))

    st.divider()
    st.header("Generate adjudicated findings and recommendations")
    st.write(
        "Run the saved human review through adjudication, "
        "worker association, and safety reasoning."
    )

    if st.button(
        "Generate adjudicated findings and recommendations",
        type="primary"
    ):
        pipeline_scripts = [
            "build_adjudicated_report.py",
            "associate_worker_findings.py",
            "safety_reasoning.py"
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
                    )
                },
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
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
                "Adjudication, worker association, and safety reasoning "
                "completed."
            )
