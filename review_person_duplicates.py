from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from person_duplicate_review import (
    MANIFEST_PATH,
    SUPPRESSION_DECISIONS,
    build_person_overlay,
    load_review_items,
    load_saved_reviews,
    save_review,
)


ROOT = Path(__file__).resolve().parent
REVIEW_PATH = (
    ROOT / "reports" / "person_duplicate_validation_final_40_reviews.json"
)

st.set_page_config(
    page_title="Person duplicate review",
    layout="wide",
)
st.title("Person duplicate validation review")
st.warning(
    "This is a descriptive review of the fixed 40-image test set. Do not use "
    "these images or judgments to tune thresholds or train models.",
    icon=":material/warning:",
)

items = load_review_items(ROOT)
reviews = load_saved_reviews(
    REVIEW_PATH,
    {item["image"] for item in items},
)
previously_documented = sum(
    item["manual_review_status"].startswith("Reviewed")
    for item in items
)
newly_reviewed = sum(
    item["image"] in reviews for item in items
    if not item["manual_review_status"].startswith("Reviewed")
)
pending = [
    item for item in items
    if item["image"] not in reviews
    and not item["manual_review_status"].startswith("Reviewed")
]

metric_columns = st.columns(3)
metric_columns[0].metric("Images in fixed set", len(items))
metric_columns[1].metric(
    "Previously documented",
    previously_documented,
)
metric_columns[2].metric("New reviews saved", newly_reviewed)
st.progress(
    (previously_documented + newly_reviewed) / len(items),
    text=(
        f"{previously_documented + newly_reviewed} of {len(items)} images "
        "have a documented review"
    ),
)
st.caption(f"Review annotations are saved separately to `{REVIEW_PATH}`.")
st.caption(f"Source manifest: `{ROOT / MANIFEST_PATH}`")

include_reviewed = st.toggle(
    "Include previously reviewed images",
    value=False,
)
queue = items if include_reviewed else pending
if not queue:
    st.success("All images in the review queue have been reviewed.")
    st.stop()

selected = st.selectbox(
    "Image to review",
    queue,
    format_func=lambda item: (
        f"{item['image']} — {item['raw_person_detections']} raw, "
        f"{item['filtered_person_detections']} retained, "
        f"{item['suppressed_person_duplicates']} suppressed"
    ),
    key=f"person_review_image_{int(include_reviewed)}",
)
saved_review = reviews.get(selected["image"], {})

st.subheader(selected["image"])
st.write(
    f"Run `{selected['run_id']}` — "
    f"{selected['raw_person_detections']} raw person detections, "
    f"{selected['filtered_person_detections']} retained, "
    f"{selected['suppressed_person_duplicates']} suppressed."
)
st.caption(
    "Green boxes are retained person detections; red boxes are detections "
    "suppressed by the configured duplicate filter. Review the image itself, "
    "not only the displayed box labels."
)

source_column, overlay_column = st.columns(2)
source_column.image(
    str(selected["source_image"]),
    caption="Original source image",
    width="stretch",
    alt=f"Original image {selected['image']}",
)
overlay_column.image(
    build_person_overlay(selected),
    caption="Retained and suppressed person detections",
    width="stretch",
    alt=f"Person detections for {selected['image']}",
)

suppressed = selected["report"].get("possible_duplicate_persons", [])
if suppressed:
    with st.expander("Suppression evidence", expanded=True):
        for index, pair in enumerate(suppressed, start=1):
            candidate = pair["candidate"]
            kept = pair["kept_detection"]
            st.write(
                f"Suppressed {index}: confidence {candidate['confidence']:.3f}; "
                f"matched retained confidence {kept['confidence']:.3f}; "
                f"IoU {pair['iou']:.3f}; containment "
                f"{pair['containment']:.3f}; area ratio "
                f"{pair['area_ratio']:.3f}; reason "
                f"`{pair['suppression_reason']}`."
            )

with st.form(f"person_review_form_{selected['image']}"):
    assessment_options = ["confident", "tentative", "uncertain"]
    previous_assessment = saved_review.get("count_assessment", "uncertain")
    if previous_assessment not in assessment_options:
        previous_assessment = "uncertain"
    count_assessment = st.selectbox(
        "Person-count assessment",
        assessment_options,
        index=assessment_options.index(previous_assessment),
        help=(
            "Use uncertain when people cannot be counted reliably due to "
            "occlusion, image quality, or ambiguity."
        ),
    )
    previous_count = saved_review.get("visible_person_count")
    visible_person_count = st.text_input(
        "Visible person count (blank if uncertain)",
        value="" if previous_count is None else str(previous_count),
    )

    suppression_reviews = []
    previous_suppression_reviews = saved_review.get(
        "suppression_reviews", []
    )
    options = [
        ("duplicate_of_kept_person", "Duplicate of retained person"),
        ("distinct_person_suppressed", "Distinct person was suppressed"),
        ("uncertain", "Uncertain"),
    ]
    for index, pair in enumerate(suppressed):
        previous_decision = (
            previous_suppression_reviews[index]
            if index < len(previous_suppression_reviews)
            else "uncertain"
        )
        option_values = [value for value, _ in options]
        if previous_decision not in SUPPRESSION_DECISIONS:
            previous_decision = "uncertain"
        decision = st.selectbox(
            f"Suppressed detection {index + 1} review",
            option_values,
            index=option_values.index(previous_decision),
            format_func=lambda value: dict(options)[value],
            help=(
                f"Candidate confidence {pair['candidate']['confidence']:.3f}; "
                f"matched retained confidence "
                f"{pair['kept_detection']['confidence']:.3f}."
            ),
            key=f"suppressed_review_{selected['image']}_{index}",
        )
        suppression_reviews.append(decision)

    reviewer_notes = st.text_area(
        "Review notes",
        value=saved_review.get("reviewer_notes", ""),
    )
    submitted = st.form_submit_button(
        "Save image review",
        type="primary",
    )

if submitted:
    count_text = visible_person_count.strip()
    if count_text and (
        not count_text.isdecimal()
        or int(count_text) < 0
    ):
        st.error("Enter a non-negative whole-number person count.")
        st.stop()
    parsed_count = int(count_text) if count_text else None
    if count_assessment == "confident" and parsed_count is None:
        st.error(
            "Enter the visible person count for a confident assessment."
        )
        st.stop()

    updated = save_review(
        REVIEW_PATH,
        items,
        reviews,
        {
            "image": selected["image"],
            "run_id": selected["run_id"],
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "count_assessment": count_assessment,
            "visible_person_count": parsed_count,
            "suppression_reviews": suppression_reviews,
            "reviewer_notes": reviewer_notes.strip(),
        },
    )
    st.session_state["person_duplicate_reviews"] = updated
    st.success("Review saved separately from the source report and dataset.")
    st.rerun()
