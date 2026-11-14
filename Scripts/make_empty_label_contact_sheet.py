from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

root = Path("data/processed/vehicle_detector/train")
label_dir = root / "labels"
image_dir = root / "images"
out_dir = Path("data/reports")
out_dir.mkdir(parents=True, exist_ok=True)

names = [
    "1237_jpg.rf.14749f3d994897cf9cc8a1156bc8ff38.txt",
    "1237_jpg.rf.c9659d3a4d688c5ba0d24ab8b994aac4.txt",
    "1328.rf.f5569a2f4b8043c05a3c80fd3a753aa7.txt",
    "448.rf.91e43c08e77058f7585d574ef84fd0c1.txt",
    "448.rf.be0060a81ff0955387a55baa28b1f1cd.txt",
    "61_jpg.rf.810d66ed31a86353a35b2bd2a6a69389.txt",
    "61_jpg.rf.9843cb24a61070b2d837e755c0654f34.txt",
]

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
thumb_w, thumb_h = 360, 260
cell_h = 300
cols = 2
rows = math.ceil(len(names) / cols)
sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), "white")
draw = ImageDraw.Draw(sheet)

for i, label_name in enumerate(names):
    stem = Path(label_name).stem
    candidates = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions and p.stem == stem
    ]

    x = (i % cols) * thumb_w
    y = (i // cols) * cell_h

    if not candidates:
        draw.text((x + 10, y + 10), "IMAGE NOT FOUND", fill="red")
        draw.text((x + 10, y + 35), stem, fill="black")
        continue

    try:
        with Image.open(candidates[0]) as im:
            im = ImageOps.contain(im.convert("RGB"), (thumb_w - 20, thumb_h - 35))
            sheet.paste(im, (x + (thumb_w - im.width) // 2, y + 5))
        draw.text((x + 8, y + thumb_h), stem[:48], fill="black")
    except Exception as exc:
        draw.text((x + 10, y + 10), str(exc)[:45], fill="red")

out = out_dir / "vehicle_empty_label_contact_sheet.jpg"
sheet.save(out, quality=90)
print("Saved:", out.resolve())
