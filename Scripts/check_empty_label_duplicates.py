
from pathlib import Path
import hashlib

ROOT = Path("data/processed/vehicle_detector")
REPORT = Path("data/reports/empty_label_duplicates_all_splits.txt")

EMPTY_IMAGES = [
    "1237_jpg.rf.14749f3d994897cf9cc8a1156bc8ff38",
    "1237_jpg.rf.c9659d3a4d688c5ba0d24ab8b994aac4",
    "1328.rf.f5569a2f4b8043c05a3c80fd3a753aa7",
    "448.rf.91e43c08e77058f7585d574ef84fd0c1",
    "448.rf.be0060a81ff0955387a55baa28b1f1cd",
    "61_jpg.rf.810d66ed31a86353a35b2bd2a6a69389",
    "61_jpg.rf.9843cb24a61070b2d837e755c0654f34",
]

SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label(label_path):
    if not label_path.exists():
        return None
    return label_path.read_text(encoding="utf-8", errors="replace").strip()


def main():
    if not ROOT.is_dir():
        print("ERROR: Dataset folder not found:", ROOT.resolve())
        return

    print("Indexing images in train, valid, and test...")
    image_index = {}
    targets = {}

    for split in SPLITS:
        image_dir = ROOT / split / "images"
        label_dir = ROOT / split / "labels"

        if not image_dir.is_dir():
            print("WARNING: Missing image directory:", image_dir)
            continue

        for image in image_dir.iterdir():
            if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
                continue

            try:
                digest = sha256(image)
                label_path = label_dir / (image.stem + ".txt")
                entry = (split, image, label_path)
                image_index.setdefault(digest, []).append(entry)

                if image.stem in EMPTY_IMAGES:
                    targets[image.stem] = (digest, entry)
            except OSError as error:
                print("Could not read", image, error)

    lines = []

    for name in EMPTY_IMAGES:
        lines.extend(["", "TARGET: " + name])

        if name not in targets:
            lines.append("  TARGET IMAGE NOT FOUND")
            continue

        digest, target_entry = targets[name]
        matches = image_index.get(digest, [])
        duplicates = [
            item for item in matches
            if item[1] != target_entry[1]
        ]

        if not duplicates:
            lines.append("  No exact duplicate found across the three splits.")
            continue

        for split, image, label_path in duplicates:
            content = read_label(label_path)

            if content is None:
                status = "LABEL FILE MISSING"
            elif not content:
                status = "LABEL EMPTY"
            else:
                status = "LABEL HAS ANNOTATIONS"

            lines.append(f"  MATCH: {split}/{image.name} — {status}")

            if content:
                lines.append("    " + content.replace("\n", "\n    "))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nReport saved to:", REPORT.resolve())
    print("Read-only check complete. No dataset files were modified.")


if __name__ == "__main__":
    main()
