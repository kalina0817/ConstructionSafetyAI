
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = ROOT / "data/raw/construction-ppe/train/labels"
IMAGE_DIR = ROOT / "data/raw/construction-ppe/train/images"
OUT_DIR = ROOT / "data/reports/candidate_label_corrections"
PROPOSALS_PATH = ROOT / "data/reports/manual_box_proposals.json"

# Classes being replaced with your reviewed proposals.
APPROVED = {
    "train/image1008_jpeg.rf.72b87d6b322cd824844053582427c688.jpg": {
        6: "no_gloves",
        7: "no_goggle",
    },
    "train/image1060_jpeg.rf.94cb69b154d29859341e2bdcb8559ade.jpg": {
        6: "no_gloves",
        7: "no_goggle",
    },
}

def yolo_box(box, class_id, image_width, image_height):
    left = float(box["left"])
    top = float(box["top"])
    right = float(box["right"])
    bottom = float(box["bottom"])

    if not (
        0 <= left < right <= image_width
        and 0 <= top < bottom <= image_height
    ):
        raise ValueError(f"Invalid box coordinates: {box}")

    xc = ((left + right) / 2) / image_width
    yc = ((top + bottom) / 2) / image_height
    width = (right - left) / image_width
    height = (bottom - top) / image_height

    return (
        f"{class_id} {xc:.8f} {yc:.8f} "
        f"{width:.8f} {height:.8f}"
    )

def main():
    data = json.loads(PROPOSALS_PATH.read_text(encoding="utf-8"))
    proposal_map = data.get("images", {})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for image_rel, class_map in APPROVED.items():
        image_name = Path(image_rel).name
        label_name = Path(image_name).with_suffix(".txt").name
        image_path = IMAGE_DIR / image_name
        source_label = LABELS_DIR / label_name

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not source_label.exists():
            raise FileNotFoundError(f"Missing source label: {source_label}")

        from PIL import Image
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        boxes = proposal_map.get(image_rel, [])
        original_lines = source_label.read_text(
            encoding="utf-8"
        ).splitlines()

        candidate_lines = []
        for line in original_lines:
            if not line.strip():
                continue
            class_id = int(line.split()[0])
            if class_id not in class_map:
                candidate_lines.append(line)

        class_counts = {}
        for class_id, class_name in class_map.items():
            matching = [
                box for box in boxes
                if box.get("class") == class_name
            ]

            if not matching:
                raise ValueError(
                    f"No saved {class_name} proposals for {image_name}"
                )

            converted = [
                yolo_box(box, class_id, image_width, image_height)
                for box in matching
            ]
            candidate_lines.extend(converted)
            class_counts[class_name] = len(converted)

        destination = OUT_DIR / label_name
        destination.write_text(
            "\n".join(candidate_lines) + "\n",
            encoding="utf-8"
        )

        summary.append({
            "image": image_name,
            "replaced_classes": class_map,
            "new_box_counts": class_counts,
            "candidate_label": str(destination.relative_to(ROOT)),
            "original_modified": False,
        })

    report = {
        "status": "CANDIDATES_GENERATED_REQUIRES_VISUAL_VERIFICATION",
        "original_labels_modified": False,
        "training_started": False,
        "files": summary,
        "note": (
            "Candidate files use saved proposals for classes 6 and 7 "
            "in both reviewed images. Visually verify candidates before "
            "approving any training."
        ),
    }

    (OUT_DIR / "candidate_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print("\nOriginal labels unchanged. Training was not started.")

if __name__ == "__main__":
    main()
