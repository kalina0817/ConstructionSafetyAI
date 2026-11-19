
import streamlit as st


st.set_page_config(
    page_title="Construction Safety AI",
    page_icon=":material/school:",
    layout="wide",
)

dashboard = st.Page(
    "dashboard_page.py",
    title="Dashboard",
    icon=":material/dashboard:",
    default=False,
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

comparison = st.Page(
    "model_comparison_page.py",
    title="Model comparison",
    icon=":material/analytics:",
)

reports = st.Page(
    "reports_page.py",
    title="Reports",
    icon=":material/description:",
)

with st.sidebar:
    st.title("Construction Safety AI")
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
        "Workspace": [
            dashboard,
            overview,
            research,
        ],
        "Prototype": [
            screening,
            comparison,
            reports,
        ],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()


