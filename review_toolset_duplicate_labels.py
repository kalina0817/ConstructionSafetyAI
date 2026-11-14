
from pathlib import Path
import json
import cv2

ROOT = Path(r"data\raw\construction-toolset")
REPORT = Path(r"data\reports\toolset_duplicate_label_comparison.json")
OUT = Path(r"data\reports\toolset_duplicate_visual_review")
OUT.mkdir(parents=True, exist_ok=True)


def draw_annotations(image_path, label_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    height, width = image.shape[:2]

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue

            try:
                class_id = int(parts[0])
                values = list(map(float, parts[1:]))

                if len(values) == 4:
                    x, y, w, h = values
                    x1 = int((x - w / 2) * width)
                    y1 = int((y - h / 2) * height)
                    x2 = int((x + w / 2) * width)
                    y2 = int((y + h / 2) * height)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image, f"class {class_id}", (max(0, x1), max(20, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

                elif len(values) >= 6 and len(values) % 2 == 0:
                    points = []
                    for i in range(0, len(values), 2):
                        points.append((
                            int(values[i] * width),
                            int(values[i + 1] * height),
                        ))
                    import numpy as np
                    cv2.polylines(
                        image, [np.array(points, dtype="int32")],
                        True, (255, 0, 0), 2
                    )
                    cv2.putText(
                        image, f"class {class_id}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                    )
            except (ValueError, IndexError):
                continue

    return image


groups = json.loads(REPORT.read_text())
shown = 0

for group in groups:
    for comparison in group["comparisons"]:
        if comparison["annotation_match"] is not False:
            continue

        pair = []
        for item in [group["reference"], comparison]:
            image_path = ROOT / item["split"] / "images" / item["image"]
            label_path = ROOT / item["split"] / "labels" / item["label"]
            image = draw_annotations(image_path, label_path)
            if image is not None:
                cv2.putText(
                    image, f"{item['split']}: {item['image'][:35]}",
                    (10, image.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
                pair.append(image)

        if len(pair) == 2:
            height = min(pair[0].shape[0], pair[1].shape[0])
            resized = [
                cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height))
                for im in pair
            ]
            canvas = cv2.hconcat(resized)
            output = OUT / f"pair_{shown + 1}.jpg"
            cv2.imwrite(str(output), canvas)
            print(output)
            shown += 1

        if shown >= 5:
            break
    if shown >= 5:
        break

print("Visual review images created:", shown)
