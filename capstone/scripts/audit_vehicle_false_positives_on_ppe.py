import json
from pathlib import Path

from ultralytics import YOLO

from construction_safety.config import MODEL_SPECS, WORKSPACE_ROOT

IMAGE_DIR = (
    WORKSPACE_ROOT
    / "data"
    / "standardized"
    / "construction-ppe"
    / "test"
    / "images"
)

OUTPUT = (
    WORKSPACE_ROOT
    / "reports"
    / "vehicle_ppe_cross_domain_audit.json"
)

CONFIDENCE = 0.70
IMAGE_SIZE = 640


def main():
    spec = MODEL_SPECS["vehicles"]
    checkpoint = WORKSPACE_ROOT / spec["checkpoint"]

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    model = YOLO(str(checkpoint))
    results = model.predict(
        source=[str(path) for path in images],
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device="cpu",
        stream=True,
        verbose=False,
    )

    records = []

    for result in results:
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                coordinates = [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ]

                detections.append({
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": round(confidence, 5),
                    "box_xyxy": coordinates,
                })

        records.append({
            "image": Path(result.path).name,
            "detections": detections,
        })

    roller_records = [
        {
            "image": record["image"],
            "roller_detections": [
                detection
                for detection in record["detections"]
                if detection["class_name"] == "Roller"
            ],
        }
        for record in records
        if any(
            detection["class_name"] == "Roller"
            for detection in record["detections"]
        )
    ]

    report = {
        "image_directory": str(IMAGE_DIR),
        "checkpoint": str(checkpoint),
        "confidence_threshold": CONFIDENCE,
        "image_size": IMAGE_SIZE,
        "images_scanned": len(images),
        "images_with_any_vehicle_detection": sum(
            bool(record["detections"]) for record in records
        ),
        "images_with_roller_predictions": len(roller_records),
        "total_roller_predictions": sum(
            len(record["roller_detections"])
            for record in roller_records
        ),
        "roller_predictions": roller_records,
        "all_predictions": records,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        key: value
        for key, value in report.items()
        if key not in {"roller_predictions", "all_predictions"}
    }, indent=2))
    print(f"\nFull report saved to: {OUTPUT}")


if __name__ == "__main__":
    main()