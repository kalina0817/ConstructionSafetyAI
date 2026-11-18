
import streamlit as st


st.set_page_config(
    page_title="Construction safety capstone",
    page_icon=":material/school:",
    layout="wide",
)

overview = st.Page(
    "capstone_home.py",
    title="Project overview",
    icon=":material/school:",
    default=True,
)

research = st.Page(
    "research_workbench.py",
    title="Research methods",
    icon=":material/science:",
)

screening = st.Page(
    "integrated_app.py",
    title="Image screening",
    icon=":material/image_search:",
)

reports = st.Page(
    "reports_page.py",
    title="Reports",
    icon=":material/description:",
)

with st.sidebar:
    st.title("Research workspace")
    st.caption("M.S. computer science capstone")
    st.caption(
        "Context-aware vision reasoning for proactive construction safety"
    )
    st.info(
        "Potential safety alerts are generated automatically. "
        "Human verification is optional. This research prototype does not "
        "confirm safety violations or replace site safety procedures."
    )

navigation = st.navigation(
    {
        "Capstone": [overview, research],
        "Prototype": [screening, reports],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()
