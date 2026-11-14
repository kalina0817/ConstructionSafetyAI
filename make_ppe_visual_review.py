from pathlib import Path
import random
import yaml
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("data/processed/construction-ppe-corrected")
OUT = Path("data/reports/visual_review")
OUT.mkdir(parents=True, exist_ok=True)

with open(ROOT / "data.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

names = config["names"]
if isinstance(names, list):
    class_names = names
else:
    class_names = [names[i] for i in sorted(names)]

image_dir = ROOT / "train" / "images"
label_dir = ROOT / "train" / "labels"

pairs = []
for label_path in sorted(label_dir.glob("*.txt")):
    image_path = next(
        (image_dir / (label_path.stem + ext)
         for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
         if (image_dir / (label_path.stem + ext)).exists()),
        None
    )
    if image_path:
        pairs.append((image_path, label_path))

if not pairs:
    raise RuntimeError("No matching PPE training images and label files found.")

random.seed(42)
selected = random.sample(pairs, min(12, len(pairs)))

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for ax, (image_path, label_path) in zip(axes, selected):
    image = cv2.imread(str(image_path))
    if image is None:
        ax.set_title(f"Could not read: {image_path.name}")
        ax.axis("off")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    ax.imshow(image)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue

        try:
            cls = int(parts[0])
            values = [float(x) for x in parts[1:]]
        except ValueError:
            continue

        if len(values) != 4 or not (0 <= cls < len(class_names)):
            continue

        xc, yc, bw, bh = values
        x = (xc - bw / 2) * w
        y = (yc - bh / 2) * h
        ax.add_patch(Rectangle(
            (x, y), bw * w, bh * h,
            fill=False, edgecolor="red", linewidth=1.5
        ))
        ax.text(
            x, max(0, y - 3), class_names[cls],
            color="white", fontsize=7,
            bbox=dict(facecolor="red", alpha=0.75, pad=1)
        )

    ax.set_title(image_path.name, fontsize=8)
    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

fig.suptitle(
    "PPE Dataset — Training Annotation Visual Review",
    fontsize=16
)
fig.tight_layout()
output = OUT / "ppe_train_visual_review.png"
fig.savefig(output, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Images available for review: {len(pairs)}")
print(f"Images displayed: {len(selected)}")
print(f"Review sheet saved to: {output}")
