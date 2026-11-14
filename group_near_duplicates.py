
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw
import re
import math

report = Path("data/reports/near_duplicate_check/near_duplicate_pairs.txt")
out_dir = Path("data/reports/near_duplicate_check/grouped_review")
out_dir.mkdir(parents=True, exist_ok=True)

text = report.read_text(encoding="utf-8")
blocks = re.split(r"(?=Distance:\s*\d+)", text)
pairs = []

for block in blocks:
    distance = re.search(r"Distance:\s*(\d+)", block)
    paths = re.findall(r"\b(?:train|valid|test):\s*(.+)", block)
    if distance and len(paths) == 2:
        pairs.append({
            "distance": int(distance.group(1)),
            "paths": [Path(p.strip()) for p in paths]
        })

# Build groups of connected candidate images.
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for pair in pairs:
    a, b = map(str, pair["paths"])
    union(a, b)

groups = defaultdict(set)
for pair in pairs:
    for p in pair["paths"]:
        groups[find(str(p))].add(str(p))

groups = [sorted(g) for g in groups.values() if len(g) > 1]
groups.sort(key=lambda g: (-len(g), g[0]))

# Make a contact sheet for each connected group.
summary = []
for number, paths in enumerate(groups, 1):
    thumb_w, thumb_h, label_h = 240, 240, 42
    cols = 4
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, name in enumerate(paths):
        p = Path(name)
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb_w - 8, thumb_h - 8))
        except Exception as e:
            print(f"Could not open {p}: {e}")
            continue

        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)
        sheet.paste(im, (x + (thumb_w - im.width) // 2, y + 4))
        draw.text((x + 5, y + thumb_h), p.parent.parent.name, fill="black")
        draw.text((x + 5, y + thumb_h + 15), p.name[:32], fill="black")

    output = out_dir / f"group_{number:02d}.jpg"
    sheet.save(output, quality=92)
    summary.append(f"Group {number}: {len(paths)} images -> {output}")

(out_dir / "group_summary.txt").write_text(
    "\n".join(summary), encoding="utf-8"
)

print(f"Candidate pairs parsed: {len(pairs)}")
print(f"Connected image groups: {len(groups)}")
print(f"Contact sheets saved in: {out_dir}")
print("Original images, labels, and splits were not modified.")
