

from pathlib import Path
import json

import streamlit as st
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "raw" / "construction-ppe"
REVIEW_FILE = ROOT / "data" / "review" / "construction_ppe_visual_reviews.json"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

st.set_page_config(page_title="Construction-PPE Review", layout="wide")
st.title("Construction-PPE Annotation Review")
st.caption(
    "Review only. Original images and labels are never edited. "
    "This app does not train a model."
)

config = yaml.safe_load(
    (DATASET / "data.yaml").read_text(encoding="utf-8")
)
names = config["names"]
REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)

if REVIEW_FILE.exists():
    reviews = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
else:
    reviews = {}

split = st.selectbox("Dataset split", ["train", "valid", "test"])
image_dir = DATASET / split / "images"
label_dir = DATASET / split / "labels"

if not image_dir.is_dir() or not label_dir.is_dir():
    st.error(f"Missing images or labels folder for {split}.")
    st.stop()

images = sorted(
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in EXTENSIONS
)

if not images:
    st.warning("No images found.")
    st.stop()

if "review_index" not in st.session_state:
    st.session_state.review_index = 0

st.session_state.review_index = min(
    st.session_state.review_index, len(images) - 1
)

index = st.number_input(
    f"Image index (0 to {len(images) - 1})",
    min_value=0,
    max_value=len(images) - 1,
    value=st.session_state.review_index,
    step=1,
)
st.session_state.review_index = int(index)

image_path = images[st.session_state.review_index]
label_path = label_dir / f"{image_path.stem}.txt"
review_key = f"{split}/{image_path.name}"

if not label_path.is_file():
    st.error(f"No matching label file: {label_path.name}")
    st.stop()

image = Image.open(image_path).convert("RGB")
draw_image = image.copy()
draw = ImageDraw.Draw(draw_image)
width, height = image.size
annotation_count = 0
problems = []

for line_number, line in enumerate(
    label_path.read_text(encoding="utf-8").splitlines(), start=1
):
    if not line.strip():
        continue

    parts = line.split()
    try:
        class_id = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
    except (ValueError, IndexError):
        problems.append(f"Invalid annotation line {line_number}")
        continue

    if not 0 <= class_id < len(names) or len(values) != 4:
        problems.append(f"Unsupported annotation line {line_number}")
        continue

    x, y, box_w, box_h = values
    left = (x - box_w / 2) * width
    top = (y - box_h / 2) * height
    right = (x + box_w / 2) * width
    bottom = (y + box_h / 2) * height

    draw.rectangle(
        [left, top, right, bottom],
        outline="red",
        width=max(2, width // 500),
    )
    draw.text(
        (left, max(0, top - 16)),
        names[class_id],
        fill="red",
        stroke_width=1,
        stroke_fill="white",
    )
    annotation_count += 1

existing = reviews.get(review_key, {})
current_status = existing.get("status", "Not reviewed")

left_col, right_col = st.columns([3, 1])

with left_col:
    st.image(
        draw_image,
        caption=image_path.name,
        width="stretch",
        alt=f"PPE annotations for {image_path.name}"
    )

with right_col:
    st.subheader("Image details")
    st.write(f"Split: {split}")
    st.write(f"Image: {image_path.name}")
    st.write(f"Annotations: {annotation_count}")
    st.write("Classes: " + ", ".join(names))

    if problems:
        st.error("\n".join(problems))

    st.write(f"Saved decision: **{current_status}**")

    status_options = ["Not reviewed", "Correct", "Needs correction"]

    selected_status = st.radio(
        "Your review decision",
        status_options,
        index=(
            status_options.index(current_status)
            if current_status in status_options
            else 0
        ),
        key=f"review_status_{split}_{image_path.name}",
    )

    notes = st.text_area(
        "Notes (optional)",
        value=existing.get("notes", ""),
        placeholder="Describe any incorrect or missing boxes or labels.",
        key=f"review_notes_{split}_{image_path.name}",
    )

    def save_current_review():
        reviews[review_key] = {
            "split": split,
            "image": image_path.name,
            "status": selected_status,
            "notes": notes,
            "annotation_count": annotation_count,
        }
        REVIEW_FILE.write_text(
            json.dumps(reviews, indent=2),
            encoding="utf-8",
        )
    st.write("DEBUG — selected status:", selected_status)
    st.write("DEBUG — current image:", image_path.name)
    if st.button("Save review decision", type="primary"):
        save_current_review()
        st.success("Review saved separately. Original annotations unchanged.")

    st.divider()
    st.write(f"Saved review records: {len(reviews)}")

    if st.button(
        "Previous image",
        disabled=st.session_state.review_index == 0,
    ):
        save_current_review()
        st.session_state.review_index -= 1
        st.rerun()

    if st.button(
        "Next image",
        disabled=st.session_state.review_index >= len(images) - 1,
    ):
        save_current_review()
        st.session_state.review_index += 1
        st.rerun()
