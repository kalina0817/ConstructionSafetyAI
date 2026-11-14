import streamlit as st


st.set_page_config(
    page_title="Construction safety capstone",
    page_icon=":material/school:",
    layout="wide",
)

overview = st.Page(
    "pages/home.py",
    title="Research overview",
    icon=":material/school:",
    default=True,
)
screening = st.Page(
    "pages/screening.py",
    title="Image screening",
    icon=":material/image_search:",
)
evaluation = st.Page(
    "pages/evaluation.py",
    title="Scenario evaluation",
    icon=":material/analytics:",
)

with st.sidebar:
    st.title("Capstone workspace")
    st.caption("M.S. computer science · research prototype")
    st.warning(
        "Findings require human verification. This is not a safety "
        "certification tool.",
        icon=":material/info:",
    )

navigation = st.navigation(
    {
        "Research": [overview, evaluation],
        "Prototype": [screening],
    },
    position="sidebar",
    expanded=True,
)
navigation.run()
