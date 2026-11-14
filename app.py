import streamlit as st


st.set_page_config(
    page_title="Construction safety capstone",
    page_icon=":material/school:",
    layout="wide",
)

overview = st.Page(
    "capstone_home.py",
    title="Research overview",
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
duplicate_review = st.Page(
    "review_person_duplicates.py",
    title="Person duplicate review",
    icon=":material/rule:",
)

with st.sidebar:
    st.title("Research workspace")
    st.caption("M.S. computer science capstone")
    st.caption(
        "Context-aware vision reasoning for proactive construction safety"
    )
    st.warning(
        "Research prototype only. Findings require qualified human review.",
        icon=":material/info:",
    )

navigation = st.navigation(
    {
        "Capstone": [overview, research],
        "Prototype": [screening, duplicate_review],
    },
    position="sidebar",
    expanded=True,
)
navigation.run()
