
from pathlib import Path
import json
import cv2

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "data/raw/construction-ppe/train/images/image1008_jpeg.rf.72b87d6b322cd824844053582427c688.jpg"
OUTPUT = ROOT / "data/reports/flagged_ppe_previews/image1008_hand_box_proposals.json"

image = cv2.imread(str(IMAGE))
if image is None:
    raise FileNotFoundError(f"Could not open image: {IMAGE}")

boxes = []
drawing = False
start = None
preview = image.copy()


def redraw():
    global preview
    preview = image.copy()
    for item in boxes:
        left, top, right, bottom = item["xyxy_pixels"]
        cv2.rectangle(preview, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            preview, item["hand"], (left, max(20, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )


def mouse(event, x, y, flags, param):
    global drawing, start, preview

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        preview = image.copy()
        redraw()
        cv2.rectangle(preview, start, (x, y), (0, 255, 0), 2)
        cv2.imshow("Draw hand boxes", preview)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        left, right = sorted([start[0], x])
        top, bottom = sorted([start[1], y])

        if right > left and bottom > top:
            boxes.append({
                "hand": f"hand_{len(boxes) + 1}",
                "xyxy_pixels": [left, top, right, bottom]
            })

        redraw()
        cv2.imshow("Draw hand boxes", preview)


cv2.namedWindow("Draw hand boxes", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Draw hand boxes", 800, 800)
cv2.setMouseCallback("Draw hand boxes", mouse)

print("Draw a rectangle around each visible hand.")
print("Press U to undo, S to save proposals, or Q to quit.")

while True:
    cv2.imshow("Draw hand boxes", preview)
    key = cv2.waitKeyEx(20)

    if key == -1:
        continue

    print("Key pressed:", key)
    key = key & 0xFF

    if key == ord("u"):
        if boxes:
            boxes.pop()
            redraw()
            cv2.imshow("Draw hand boxes", preview)

    elif key == ord("s"):
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "image": str(IMAGE.relative_to(ROOT)),
            "purpose": "Proposed hand-region boxes for manual review",
            "coordinates_format": "xyxy_pixels",
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "proposals": boxes,
            "original_labels_modified": False,
            "training_started": False
        }
        OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved {len(boxes)} proposed boxes to: {OUTPUT}")
        break

    elif key == ord("q"):
        print("Exited without saving.")
        break

cv2.destroyAllWindows()
