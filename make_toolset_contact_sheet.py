from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv, math

ROOT = Path(".")
DATA = ROOT / "data" / "raw" / "construction-toolset"
REPORT = ROOT / "data" / "reports"
OUT = REPORT / "toolset_flagged_annotation_contact_sheet.jpg"

AUDIT = REPORT / "toolset_duplicate_optimal_match_audit.csv"

def locate_image(filename):
    matches = list(DATA.glob(f"*/images/{filename}"))
    if not matches:
        matches = list(DATA.glob(f"*/{filename}"))
    return matches[0] if len(matches) == 1 else None

def locate_label(image_path):
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")

def draw_annotations(img, label_path):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    if not label_path.exists():
        draw.text((8, 8), "LABEL FILE MISSING", fill="red")
        return img

    w, h = img.size
    for line in label_path.read_text().splitlines():
        values = line.split()
        try:
            cls = values[0]
            nums = list(map(float, values[1:]))
            if len(nums) == 4:
                cx, cy, bw, bh = nums
                x1, y1 = (cx-bw/2)*w, (cy-bh/2)*h
                x2, y2 = (cx+bw/2)*w, (cy+bh/2)*h
                draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
                draw.text((x1, max(0, y1-16)), f"class {cls}", fill="red")
            elif len(nums) >= 6 and len(nums) % 2 == 0:
                points = [(nums[i]*w, nums[i+1]*h)
                          for i in range(0, len(nums), 2)]
                draw.polygon(points, outline="lime", width=3)
                draw.text(points[0], f"class {cls}", fill="lime")
            else:
                draw.text((8, 8), "UNRECOGNIZED LABEL FORMAT", fill="red")
        except (ValueError, IndexError):
            draw.text((8, 8), "INVALID LABEL LINE", fill="red")
    return img

if not AUDIT.exists():
    raise SystemExit(f"Audit CSV not found: {AUDIT}")

with AUDIT.open(newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

flagged = [r for r in rows
           if r.get("optimal_match_status") != "HIGH_BOX_OVERLAP"]

if not flagged:
    raise SystemExit("No flagged pairs found in the audit CSV.")

cards = []
for row in flagged:
    for side in ("reference_image", "comparison_image"):
        name = row.get(side, "")
        path = locate_image(name)
        if path is None:
            print(f"SKIPPED (image not uniquely found): {name}")
            continue
        label_path = locate_label(path)
        image = draw_annotations(Image.open(path), label_path)
        image.thumbnail((480, 300))
        cards.append((name, row.get("optimal_match_status", ""), image.copy()))

if not cards:
    raise SystemExit("No images found. Dataset folder structure may differ.")

cell_w, cell_h = 500, 350
cols = 2
rows_count = math.ceil(len(cards) / cols)
sheet = Image.new("RGB", (cols*cell_w, rows_count*cell_h), "white")
d = ImageDraw.Draw(sheet)

for i, (name, status, img) in enumerate(cards):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h
    d.text((x+8, y+5), name[:68], fill="black")
    d.text((x+8, y+23), status, fill="darkred")
    sheet.paste(img, (x+8, y+45))

REPORT.mkdir(parents=True, exist_ok=True)
sheet.save(OUT, quality=90)
print(f"Created: {OUT}")
print(f"Images included: {len(cards)}")
print("Red outlines = bounding boxes; green outlines = polygons.")
print("Dataset images and labels were not modified.")
