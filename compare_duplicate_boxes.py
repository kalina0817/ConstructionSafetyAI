
from pathlib import Path
import hashlib

import cv2
import yaml
import numpy as np

ROOT = Path("data/raw/construction-vehicle-detection")
OUTPUT = Path("data/reports/duplicate_box_comparisons")

PAIRS = [
    (
        "forklift-1997-_png.rf.cc91361d7eaab5d3fddf2198fd409736.jpg",
        "forklift-108-_png.rf.1f55138c3503a7db8b0a9a75c5f7b289.jpg",
    ),
    (
        "Rsafee_png.rf.ee704438c6aff1e23e9502a43ba3a558.jpg",
        "Lsafee_png.rf.951d8e853be65483d0c8c7485852fec3.jpg",
    ),
    (
        "Rsafee_png.rf.f378898e81ca2b3475ef85d9afdf8d5c.jpg",
        "Lsafee_png.rf.4f127ae802b6d6ef33887ac15a44caf3.jpg",
    ),
]


def find_class_name(names, class_id):
    if isinstance(names, list):
        return names[class_id] if class_id < len(names) else str(class_id)
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    return str(class_id)


def draw_annotations(image_path, label_path, names):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    if not label_path.exists():
        raise FileNotFoundError(f"Missing label: {label_path}")

    lines = label_path.read_text(encoding="utf-8-sig").splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue

        x1 = round((x - w / 2) * width)
        y1 = round((y - h / 2) * height)
        x2 = round((x + w / 2) * width)
        y2 = round((y + h / 2) * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        class_name = find_class_name(names, class_id)
        cv2.putText(
            image,
            class_name,
            (max(5, x1), max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    return image


def fit_image(image, max_width=1000, max_height=750):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    return cv2.resize(
        image,
        (round(w * scale), round(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def main():
    config_path = ROOT / "data.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    names = config["names"]

    images_dir = ROOT / "train" / "images"
    labels_dir = ROOT / "train" / "labels"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for index, (name_a, name_b) in enumerate(PAIRS, start=1):
        path_a = images_dir / name_a
        path_b = images_dir / name_b

        image_a = draw_annotations(
            path_a, labels_dir / f"{path_a.stem}.txt", names
        )
        image_b = draw_annotations(
            path_b, labels_dir / f"{path_b.stem}.txt", names
        )

        image_a = fit_image(image_a)
        image_b = fit_image(image_b)

        canvas_height = max(image_a.shape[0], image_b.shape[0]) + 100
        canvas_width = image_a.shape[1] + image_b.shape[1] + 30
        canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

        x_a = 10
        x_b = image_a.shape[1] + 20
        y_a = 80 + (canvas_height - 80 - image_a.shape[0]) // 2
        y_b = 80 + (canvas_height - 80 - image_b.shape[0]) // 2

        canvas[y_a:y_a + image_a.shape[0], x_a:x_a + image_a.shape[1]] = image_a
        canvas[y_b:y_b + image_b.shape[0], x_b:x_b + image_b.shape[1]] = image_b

        cv2.putText(
            canvas, f"PAIR {index} - IMAGE A", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2
        )
        cv2.putText(
            canvas, f"PAIR {index} - IMAGE B", (x_b, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2
        )

        output_path = OUTPUT / f"duplicate_pair_{index}.jpg"
        cv2.imwrite(str(output_path), canvas)
        print(f"Created: {output_path}")

    print("\nFinished. Original images and labels were not changed.")


if __name__ == "__main__":
    main()
