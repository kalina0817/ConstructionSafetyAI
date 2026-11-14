
from pathlib import Path
from collections import Counter
import csv
import yaml
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "construction-ppe"
OUT = ROOT / "data" / "review" / "automated_ppe_audit"
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load((DATA / "data.yaml").read_text(encoding="utf-8"))
names = cfg["names"]
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class_counts = Counter()
flags = []
records = []
contact_items = Counter()
thumbs = {}
total_boxes = 0

for split in ("train", "valid", "test"):
    img_dir = DATA / split / "images"
    lbl_dir = DATA / split / "labels"

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue

        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            flags.append([split, img_path.name, "", "Missing label file"])
            continue

        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception as e:
            flags.append([split, img_path.name, "", f"Unreadable image: {e}"])
            continue

        boxes = []
        for line_no, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue

            parts = line.split()
            try:
                cid = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except (ValueError, IndexError):
                flags.append([split, img_path.name, line_no, "Invalid annotation"])
                continue

            if not 0 <= cid < len(names) or len(vals) != 4:
                flags.append([split, img_path.name, line_no, "Unexpected class or format"])
                continue

            x, y, w, h = vals
            class_counts[(split, cid)] += 1
            total_boxes += 1

            if not all(0 <= v <= 1 for v in vals):
                flags.append([split, img_path.name, line_no, "Coordinates outside 0–1"])
                continue
            if w <= 0 or h <= 0 or x-w/2 < 0 or x+w/2 > 1 or y-h/2 < 0 or y+h/2 > 1:
                flags.append([split, img_path.name, line_no, "Invalid box size or boundary"])
                continue

            area = w * h
            if area < 0.0005:
                flags.append([split, img_path.name, line_no, f"Extremely small box ({area:.6f} image area)"])
            elif area < 0.002:
                flags.append([split, img_path.name, line_no, f"Very small box ({area:.6f} image area)"])

            boxes.append((cid, x, y, w, h))

        # Save thumbnails only for images with flagged boxes.
        if any(f[0] == split and f[1] == img_path.name for f in flags):
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((240, 160))
                    draw = ImageDraw.Draw(im)
                    for cid, x, y, w, h in boxes:
                        x1 = int((x-w/2) * im.width)
                        y1 = int((y-h/2) * im.height)
                        x2 = int((x+w/2) * im.width)
                        y2 = int((y+h/2) * im.height)
                        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
                        draw.text((x1, max(0, y1-12)), names[cid], fill="yellow",
                                  stroke_width=1, stroke_fill="black")
                    thumbs[(split, img_path.name)] = (im.copy(), len(boxes))
            except Exception:
                pass

# Class counts by split
with (OUT / "class_counts.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "class_id", "class_name", "annotation_count"])
    for split in ("train", "valid", "test"):
        for cid, name in enumerate(names):
            writer.writerow([split, cid, name, class_counts[(split, cid)]])

with (OUT / "flagged_annotations.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "image", "line", "reason"])
    writer.writerows(flags)

# Contact sheets contain only images flagged by automated checks.
flagged_keys = sorted(thumbs)
for start in range(0, len(flagged_keys), 60):
    batch = flagged_keys[start:start+60]
    sheet = Image.new("RGB", (4*250, 15*190), "white")
    draw = ImageDraw.Draw(sheet)

    for i, key in enumerate(batch):
        thumb, count = thumbs[key]
        x = (i % 4) * 250
        y = (i // 4) * 190
        sheet.paste(thumb, (x+5, y+20))
        draw.text((x+5, y+3), f"{key[0]}: {key[1]}"[:38], fill="black")
        draw.text((x+5, y+183), f"Boxes: {count}", fill="black")

    sheet.crop((0, 0, 1000, ((len(batch)+3)//4)*190)).save(
        OUT / f"flagged_contact_sheet_{start//60+1:02d}.jpg"
    )

with (OUT / "audit_summary.txt").open("w", encoding="utf-8") as f:
    f.write(f"Images with automated flags: {len(thumbs)}\n")
    f.write(f"Flagged annotation/issues rows: {len(flags)}\n")
    f.write(f"Valid boxes counted: {total_boxes}\n")
    f.write("No original images or labels were modified.\n")
    f.write("Flags indicate items worth checking, not confirmed annotation errors.\n")

print("AUDIT COMPLETE")
print("Images with flags:", len(thumbs))
print("Flagged issues:", len(flags))
print("Annotations counted:", total_boxes)
print("Results folder:", OUT)
print("Original dataset unchanged. No training started.")
