from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

root = Path.cwd()
images_dir = root / "data/raw/construction-ppe/train/images"
original_labels_dir = root / "data/raw/construction-ppe/train/labels"
candidate_dir = root / "data/reports/candidate_label_corrections"
output_dir = root / "data/reports/candidate_label_previews"

# Confirmed class IDs from the dataset.
names = {
    0: "Person",
    4: "helmet",
    6: "no_gloves",
    7: "no_goggle",
    9: "no_vest",
    10: "vest",
}

report = json.loads((candidate_dir / "candidate_summary.json").read_text(encoding="utf-8"))
output_dir.mkdir(parents=True, exist_ok=True)

def draw_labels(image, label_path, draw, font):
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    width, height = image.size
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        label = names.get(cls, f"class_{cls}")
        color = "#ff3030" if cls in (6, 7) else "#00c853"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, max(0, y1 - 18)), label, fill=color, font=font)

for item in report["files"]:
    image_name = item["image"]
    stem = Path(image_name).stem
    image_path = images_dir / image_name
    original_label = original_labels_dir / f"{stem}.txt"
    candidate_label = candidate_dir / f"{stem}.txt"

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    # Show candidate boxes in the main preview.
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw_labels(image, candidate_label, draw, font)

    image.save(output_dir / f"{stem}_candidate.jpg", quality=95)

    # Also save the original-label view for comparison.
    original = Image.open(image_path).convert("RGB")
    original_draw = ImageDraw.Draw(original)
    draw_labels(original, original_label, original_draw, font)
    original.save(output_dir / f"{stem}_original.jpg", quality=95)

print(f"Previews saved to: {output_dir}")
print("Files ending in _candidate.jpg show candidate labels.")
print("Files ending in _original.jpg show original labels.")
print("No original images or labels were changed.")
