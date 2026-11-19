import streamlit as st


st.set_page_config(
    page_title="ConstructionSafetyAI",
    page_icon=":material/construction:",
    layout="wide",
)

overview = st.Page(
    "pages/home.py",
    title="Project Overview",
    icon=":material/home:",
    default=True,
)

screening = st.Page(
    "pages/screening.py",
    title="Image Screening",
    icon=":material/image_search:",
)

evaluation = st.Page(
    "pages/evaluation.py",
    title="Scenario Evaluation",
    icon=":material/analytics:",
)

with st.sidebar:
    st.title("ConstructionSafetyAI")
    st.caption("Construction site image screening")
    st.warning(
        "Automated image screening only. Results are not safety "
        "certification or proof that a site is safe.",
        icon=":material/info:",
    )

navigation = st.navigation(
    {
        "Project": [overview, evaluation],
        "Screening": [screening],
    },
    position="sidebar",
    expanded=True,
)

navigation.run()