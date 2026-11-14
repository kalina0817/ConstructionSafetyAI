
from pathlib import Path
import shutil
import yaml

SOURCE = Path("data/processed/construction-toolset-detection")
OUTPUT = Path("data/processed/construction-toolset-safety")
SPLITS = ("train", "valid", "test")

# Original class IDs mapped to their selected class names.
SELECTED = {
    0: "Excavator",
    10: "gloves",
    22: "safety glass",
    23: "safety headphones",
    24: "safety helmet",
    25: "safety mask",
    26: "safety_vest",
    29: "shovel",
    33: "wheel barrow",
}


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Source not found: {SOURCE}")

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT}\n"
            "Please use an empty output folder."
        )

    config_path = SOURCE / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_names = config["names"]

    # Check that the selected IDs match the source dataset.
    for old_id, expected_name in SELECTED.items():
        if source_names[old_id] != expected_name:
            raise ValueError(
                f"Class mismatch at ID {old_id}: "
                f"expected {expected_name!r}, got {source_names[old_id]!r}"
            )

    remap = {
        old_id: new_id
        for new_id, old_id in enumerate(SELECTED)
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in SPLITS:
        src_images = SOURCE / split / "images"
        src_labels = SOURCE / split / "labels"
        dst_images = OUTPUT / split / "images"
        dst_labels = OUTPUT / split / "labels"

        if not src_images.is_dir() or not src_labels.is_dir():
            raise FileNotFoundError(f"Missing images or labels for {split}")

        image_files = [
            p for p in src_images.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
        label_files = list(src_labels.glob("*.txt"))

        image_stems = {p.stem for p in image_files}
        label_stems = {p.stem for p in label_files}

        if image_stems != label_stems:
            raise ValueError(f"Image/label mismatch in {split}")

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        kept_count = 0
        skipped_count = 0
        empty_count = 0

        for image in image_files:
            shutil.copy2(image, dst_images / image.name)

        for label_file in sorted(label_files):
            kept = []

            for line_number, line in enumerate(
                label_file.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue

                values = line.split()
                if len(values) != 5:
                    raise ValueError(
                        f"Unexpected annotation in {label_file}, "
                        f"line {line_number}"
                    )

                old_id = int(float(values[0]))

                if old_id not in remap:
                    skipped_count += 1
                    continue

                values[0] = str(remap[old_id])
                kept.append(" ".join(values))
                kept_count += 1

            if not kept:
                empty_count += 1

            (dst_labels / label_file.name).write_text(
                "\n".join(kept) + ("\n" if kept else ""),
                encoding="utf-8",
            )

        print(f"\n{split.upper()}")
        print("Images copied:", len(image_files))
        print("Selected boxes:", kept_count)
        print("Other-class boxes skipped:", skipped_count)
        print("Empty output labels:", empty_count)

    output_config = {
        "path": str(OUTPUT.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(SELECTED),
        "names": list(SELECTED.values()),
    }

    (OUTPUT / "data.yaml").write_text(
        yaml.safe_dump(output_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("\nSafety-focused dataset created.")
    print("Output:", OUTPUT.resolve())
    print("\nNew class mapping:")

    for new_id, name in enumerate(SELECTED.values()):
        print(f"  {new_id}: {name}")

    print("\nOriginal datasets were not modified.")


if __name__ == "__main__":
    main()
