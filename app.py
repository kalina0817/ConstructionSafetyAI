
import streamlit as st


st.set_page_config(
    page_title="Construction Safety AI",
    page_icon=":material/school:",
    layout="wide",
)

st.html("""
<style>
button[kind="primary"]:enabled,
button[kind="primary"]:enabled p { color: #211A10; }
</style>
""")

dashboard = st.Page(
    "dashboard_page.py",
    title="What the app checks",
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
    title="How we test",
    icon=":material/science:",
)

screening = st.Page(
    "integrated_app.py",
    title="Check a photo",
    icon=":material/image_search:",
)

comparison = st.Page(
    "model_comparison_page.py",
    title="Compare results",
    icon=":material/analytics:",
)

reports = st.Page(
    "reports_page.py",
    title="Saved reports",
    icon=":material/description:",
)

with st.sidebar:
    st.subheader("Construction Safety AI")
    st.caption("Jobsite Photo Review")

navigation = st.navigation(
    {
        "About the project": [overview, dashboard],
        "Photo checks": [screening, reports],
        "Testing and results": [research, comparison],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()


