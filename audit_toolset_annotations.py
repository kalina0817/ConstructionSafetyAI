
from pathlib import Path

root = Path(r"data\raw\construction-toolset")

for split in ["train", "valid", "test"]:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"

    images = sum(1 for f in image_dir.iterdir() if f.is_file())
    labels = len(list(label_dir.glob("*.txt")))

    print(f"{split}: images={images}, labels={labels}")
