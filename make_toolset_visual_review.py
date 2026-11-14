from pathlib import Path
import random
import yaml
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

ROOT = Path("data/processed/construction-toolset-deduplicated")
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
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

pairs = []
for label_path in sorted(label_dir.glob("*.txt")):
    image_path = next(
        (image_dir / (label_path.stem + ext)
         for ext in extensions
         if (image_dir / (label_path.stem + ext)).exists()),
        None
    )
    if image_path:
        pairs.append((image_path, label_path))

if not pairs:
    raise RuntimeError("No matching toolset training images and labels found.")

random.seed(42)
selected = random.sample(pairs, min(12, len(pairs)))

fig, axes = plt.subplots(3, 4, figsize=(22, 17))
axes = axes.flatten()
counts = {"boxes": 0, "polygons": 0, "skipped": 0}

for ax, (image_path, label_path) in zip(axes, selected):
    image = cv2.imread(str(image_path))
    if image is None:
        ax.set_title(f"Unreadable: {image_path.name}")
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
            values = [float(v) for v in parts[1:]]
        except ValueError:
            counts["skipped"] += 1
            continue

        if not 0 <= cls < len(class_names):
            counts["skipped"] += 1
            continue

        if len(values) == 4:
            xc, yc, bw, bh = values
            x, y = (xc - bw / 2) * w, (yc - bh / 2) * h
            ax.add_patch(Rectangle(
                (x, y), bw * w, bh * h,
                fill=False, edgecolor="red", linewidth=1.5
            ))
            counts["boxes"] += 1

        elif len(values) >= 6 and len(values) % 2 == 0:
            points = [
                (values[i] * w, values[i + 1] * h)
                for i in range(0, len(values), 2)
            ]
            ax.add_patch(Polygon(
                points, closed=True, fill=False,
                edgecolor="yellow", linewidth=1.2
            ))
            x, y = points[0]
            counts["polygons"] += 1

        else:
            counts["skipped"] += 1
            continue

        ax.text(
            max(0, x), max(0, y - 3), class_names[cls],
            color="black", fontsize=7,
            bbox=dict(facecolor="yellow", alpha=0.85, pad=1)
        )

    ax.set_title(image_path.name, fontsize=8)
    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

fig.suptitle(
    "Toolset Dataset — Bounding Boxes (Red) and Polygons (Yellow)",
    fontsize=15
)
fig.tight_layout()

output = OUT / "toolset_train_visual_review.png"
fig.savefig(output, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Matching training images: {len(pairs)}")
print(f"Images selected: {len(selected)}")
print(f"Bounding boxes displayed: {counts['boxes']}")
print(f"Polygons displayed: {counts['polygons']}")
print(f"Annotations skipped: {counts['skipped']}")
print(f"Review sheet saved to: {output}")
