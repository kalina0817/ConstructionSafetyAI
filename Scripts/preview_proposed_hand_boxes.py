
from pathlib import Path
import cv2

root = Path(__file__).resolve().parent.parent

image_path = root / (
    "data/raw/construction-ppe/train/images/"
    "image1008_jpeg.rf.72b87d6b322cd824844053582427c688.jpg"
)

output_path = root / (
    "data/reports/flagged_ppe_previews/"
    "image1008_proposed_hand_boxes.jpg"
)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

# Proposed boxes only: (left, top, right, bottom)
boxes = [
    (390, 510, 420, 540),  # Right hand, viewer's left
    (550, 540, 580, 570),  # Left hand, viewer's right
]

for i, (left, top, right, bottom) in enumerate(boxes, start=1):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(
        image, f"Proposed hand {i}", (left, max(20, top - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
    )

cv2.imwrite(str(output_path), image)
print(f"Preview created: {output_path}")
print("Original image and label files were not changed.")
