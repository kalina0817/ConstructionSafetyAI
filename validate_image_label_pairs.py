from pathlib import Path

dataset = Path("data/processed/construction-ppe-corrected")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
problems = []

for split in ["train", "valid", "test"]:
    image_dir = dataset / split / "images"
    label_dir = dataset / split / "labels"

    images = {
        p.stem for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    }
    labels = {p.stem for p in label_dir.glob("*.txt")}

    missing_labels = sorted(images - labels)
    missing_images = sorted(labels - images)

    print(f"\n{split}:")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Images missing labels:", len(missing_labels))
    print("Labels missing images:", len(missing_images))

    problems.extend(
        f"{split}: image missing label: {name}"
        for name in missing_labels
    )
    problems.extend(
        f"{split}: label missing image: {name}"
        for name in missing_images
    )

print("\nTotal pairing problems:", len(problems))
for problem in problems[:20]:
    print(problem)