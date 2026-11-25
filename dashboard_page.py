from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent

MODEL_PATHS = {
    "PPE": ROOT / "runs/detect/runs/detect/ppe_yolo11n_postcorrection_15ep/weights/best.pt",
    "Toolset": ROOT / "models/toolset35_final_model/toolset35_yolo11n_best.pt",
    "Vehicles": ROOT / "runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt",
}

st.title("What the app checks")
st.write("The app looks for three groups of objects in a site photo.")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Worker and gear labels", 11)

with col2:
    st.metric("Tool and equipment categories", 35)

with col3:
    st.metric("Vehicle categories", 10)

st.divider()

st.subheader("Files used by YOLO11n")

st.write(
    "These are the model files the application checks before running "
    "image analysis."
)

for model_name, model_path in MODEL_PATHS.items():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(f"**{model_name} model**")
        with st.expander("File location"):
            st.code(str(model_path.relative_to(ROOT)))

    with col2:
        if model_path.is_file():
            st.success("Model file found")
        else:
            st.error("Model file missing")

st.divider()

st.caption("A worker’s hands or feet may be hidden. If you cannot see the gear clearly, check on site.")
