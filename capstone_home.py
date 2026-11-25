from pathlib import Path

import streamlit as st

STUDY = Path(__file__).resolve().parent / "research" / "worker_machine_blind_12"
example = STUDY / "evaluation/yolo11n_rules_baseline/Photo_01/unified_detections.jpg"

st.html("""
<style>
.st-key-project_intro {
    background: linear-gradient(110deg, #201c16, #121313 56%);
    border: 1px solid #484136;
    border-radius: 18px;
    padding: clamp(1rem, 2.5vw, 2.4rem);
}
.st-key-project_intro h1 {
    font-size: clamp(2.6rem, 4vw, 4.4rem);
    line-height: 1.04;
    letter-spacing: -0.035em;
}
.st-key-project_intro img { border-radius: 10px; }
</style>
""")

with st.container(key="project_intro"):
    intro, picture = st.columns([5, 6], gap="large", vertical_alignment="center")
    with intro:
        st.caption("Jobsite Photo Review")
        st.title("Construction Safety AI")
        st.write("Review a jobsite photo for two things that matter to worker safety: "
                 "what protective gear each person appears to be wearing and where people are standing near machinery.")
        if st.button("Check a site photo", type="primary", icon=":material/image_search:"):
            st.switch_page("integrated_app.py")
    with picture:
        if example.is_file():
            st.image(str(example), width="stretch", alt="Worker in an excavator cab with boxes around objects found by the app")
            st.caption("The boxes show where the app found objects in the photo.")

st.header("From photo to safety report")
setup, report = st.columns(2, gap="large")
with setup, st.container(border=True):
    st.subheader("What the app checks")
    st.write("Upload a photo and choose how the app finds objects and how it checks for risks. "
             "Run the check once to get a labeled photo and a report.")
    st.write("For each worker it finds, the app checks five items: a hard hat, a safety vest, "
             "gloves, boots, and goggles. It also looks at the worker’s position around machines.")
with report, st.container(border=True):
    st.subheader("What you can learn")
    st.write("Each finding explains what raised the concern, what remains unclear, "
             "and what to check on site. You can download the report to share those details.")
    st.write("For example, if a tool hides someone’s hands, the report can mark their gloves "
             "as “cannot tell.” A clearer photo or a check on site can settle the question.")
