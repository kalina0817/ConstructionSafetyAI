
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MANIFEST_PATH = Path("reports/person_duplicate_validation_final_40.csv")
REVIEW_SCHEMA_VERSION = "1.0"
SUPPRESSION_DECISIONS = {
    "duplicate_of_kept_person",
    "distinct_person_suppressed",
    "uncertain",
}

REQUIRED_MANIFEST_COLUMNS = {
    "image",
    "run_id",
    "raw_person_detections",
    "filtered_person_detections",
    "suppressed_person_duplicates",
    "manual_review_status",
    "manual_review_notes",
}

SOURCE_IMAGE_FALLBACKS = (
    Path("data/raw/construction-ppe/test/images"),
    Path("data/standardized/construction-ppe/test/images"),
    Path("data/processed/construction-ppe/test/images"),
    Path("data/processed/ppe_detector/test/images"),
)


def resolve_source_image(project_root, source_image, image_name):
    source_image = source_image.resolve()

    if source_image.is_file():
        return source_image

    for relative_directory in SOURCE_IMAGE_FALLBACKS:
        candidate = (
            project_root / relative_directory / image_name
        ).resolve()

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Source image not found for {image_name}. "
        f"Original path: {source_image}. "
        "Checked the configured PPE image fallback directories."
    )


def load_review_items(project_root):
    project_root = Path(project_root).resolve()
    manifest_path = project_root / MANIFEST_PATH
    runs_dir = (project_root / "reports" / "inference_runs").resolve()

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not REQUIRED_MANIFEST_COLUMNS.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Review manifest is missing required columns: "
                f"{sorted(REQUIRED_MANIFEST_COLUMNS)}"
            )

        items = []
        seen_images = set()

        for line_number, row in enumerate(reader, start=2):
            image_name = row["image"].strip()
            run_id = row["run_id"].strip()

            if not image_name or not run_id:
                raise ValueError(
                    f"Review manifest row {line_number} has an empty image "
                    "or run ID."
                )

            if Path(image_name).name != image_name:
                raise ValueError(
                    f"Review manifest row {line_number} has an invalid image "
                    f"name: {image_name!r}"
                )

            if image_name in seen_images:
                raise ValueError(
                    f"Duplicate image in review manifest: {image_name!r}"
                )

            seen_images.add(image_name)

            run_folder = (runs_dir / run_id).resolve()
            if run_folder.parent != runs_dir:
                raise ValueError(
                    f"Review manifest row {line_number} has an invalid run "
                    f"ID: {run_id!r}"
                )

            report_path = run_folder / "detections.json"
            overlay_path = run_folder / "unified_detections.jpg"

            if not report_path.is_file():
                raise FileNotFoundError(
                    f"Detection report not found for {image_name}: "
                    f"{report_path}"
                )

            if not overlay_path.is_file():
                raise FileNotFoundError(
                    f"Detection overlay not found for {image_name}: "
                    f"{overlay_path}"
                )

            with report_path.open("r", encoding="utf-8") as report_file:
                report = json.load(report_file)

            source_image = Path(report.get("input_image", ""))
            if not source_image.is_absolute():
                source_image = project_root / source_image

            if source_image.name != image_name:
                raise ValueError(
                    f"Detection report {report_path} references "
                    f"{source_image.name!r}, expected {image_name!r}."
                )

            source_image = resolve_source_image(
                project_root,
                source_image,
                image_name,
            )

            for column, report_key in (
                ("raw_person_detections", "raw_person_detection_count"),
                ("filtered_person_detections", "filtered_person_detection_count"),
                ("suppressed_person_duplicates", "possible_duplicate_person_count"),
            ):
                if int(row[column]) != report.get(report_key):
                    raise ValueError(
                        f"Manifest and detection report disagree for "
                        f"{image_name}: {column}."
                    )

            possible_duplicates = report.get("possible_duplicate_persons")
            if not isinstance(possible_duplicates, list):
                raise ValueError(
                    f"Invalid possible_duplicate_persons in {report_path}."
                )

            if len(possible_duplicates) != int(
                row["suppressed_person_duplicates"]
            ):
                raise ValueError(
                    f"Suppressed detection count does not match for "
                    f"{image_name}."
                )

            items.append({
                "image": image_name,
                "run_id": run_id,
                "source_image": source_image,
                "report_path": report_path,
                "overlay_path": overlay_path,
                "report": report,
                "manual_review_status": row["manual_review_status"].strip(),
                "manual_review_notes": row["manual_review_notes"].strip(),
                "raw_person_detections": int(
                    row["raw_person_detections"]
                ),
                "filtered_person_detections": int(
                    row["filtered_person_detections"]
                ),
                "suppressed_person_duplicates": int(
                    row["suppressed_person_duplicates"]
                ),
            })

    if not items:
        raise ValueError(f"No images found in review manifest: {manifest_path}")

    return items


def load_saved_reviews(review_path, valid_images):
    review_path = Path(review_path)
    if not review_path.exists():
        return {}

    with review_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported review schema version in {review_path}."
        )

    saved = {}

    for review in payload.get("reviews", []):
        image_name = review.get("image")

        if image_name not in valid_images:
            raise ValueError(
                f"Saved review references an image outside the manifest: "
                f"{image_name!r}"
            )

        if image_name in saved:
            raise ValueError(
                f"Duplicate saved review for {image_name!r}."
            )

        saved[image_name] = review

    return saved


def save_review(review_path, items, reviews, review):
    image_name = review.get("image")
    valid_images = {item["image"] for item in items}

    if image_name not in valid_images:
        raise ValueError(
            f"Review image is not in the manifest: {image_name!r}"
        )

    if review.get("count_assessment") not in {
        "confident",
        "tentative",
        "uncertain",
    }:
        raise ValueError("Invalid person-count assessment.")

    visible_count = review.get("visible_person_count")

    if visible_count is not None and (
        not isinstance(visible_count, int) or visible_count < 0
    ):
        raise ValueError(
            "Visible person count must be a non-negative integer or null."
        )

    if (
        review["count_assessment"] == "confident"
        and visible_count is None
    ):
        raise ValueError(
            "A confident person-count assessment requires a visible count."
        )

    expected_suppression_count = next(
        item["suppressed_person_duplicates"]
        for item in items
        if item["image"] == image_name
    )

    suppression_reviews = review.get("suppression_reviews", [])

    if len(suppression_reviews) != expected_suppression_count:
        raise ValueError(
            "Every suppressed person detection must have one review decision."
        )

    if any(
        decision not in SUPPRESSION_DECISIONS
        for decision in suppression_reviews
    ):
        raise ValueError("Invalid suppressed-detection review decision.")

    updated_reviews = dict(reviews)
    updated_reviews[image_name] = review

    ordered_reviews = [
        updated_reviews[item["image"]]
        for item in items
        if item["image"] in updated_reviews
    ]

    payload = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source_manifest": str(MANIFEST_PATH),
        "purpose": (
            "Descriptive audit of the fixed duplicate-filter validation set; "
            "not a threshold-tuning or model-training set."
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviews": ordered_reviews,
    }

    review_path = Path(review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = review_path.with_suffix(
        review_path.suffix + ".tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(review_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return updated_reviews


def build_person_overlay(item):
    report = item["report"]
    image = Image.open(item["source_image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    retained = [
        detection
        for detection in report.get("analysis_detections", [])
        if detection.get("class_name", "").strip().lower() == "person"
    ]

    for index, detection in enumerate(retained, start=1):
        box = tuple(int(value) for value in detection["box_xyxy"])
        label = f"Kept {index} ({detection['confidence']:.2f})"

        draw.rectangle(box, outline=(0, 170, 0), width=3)
        draw.text(
            (box[0], max(0, box[1] - 14)),
            label,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 100, 0),
        )

    for index, pair in enumerate(
        report.get("possible_duplicate_persons", []),
        start=1,
    ):
        detection = pair["candidate"]
        box = tuple(int(value) for value in detection["box_xyxy"])
        label = f"Suppressed {index} ({detection['confidence']:.2f})"

        draw.rectangle(box, outline=(220, 0, 0), width=3)
        draw.text(
            (box[0], max(0, box[1] - 14)),
            label,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(150, 0, 0),
        )

    return image
