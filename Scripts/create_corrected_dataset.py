
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "raw" / "construction-ppe"
DEST = ROOT / "data" / "processed" / "construction-ppe-corrected"
CANDIDATES = ROOT / "data" / "reports" / "candidate_label_corrections"
APPROVAL = CANDIDATES / "visual_approval.json"

APPROVED_FILES = [
    "image1008_jpeg.rf.72b87d6b322cd824844053582427c688.txt",
    "image1060_jpeg.rf.94cb69b154d29859341e2bdcb8559ade.txt",
]


def main():
    # Safety checks: do not overwrite an existing destination.
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    if DEST.exists():
        raise FileExistsError(
            f"Destination already exists: {DEST}\n"
            "Nothing was overwritten. Choose a new destination folder."
        )

    if not APPROVAL.is_file():
        raise FileNotFoundError(f"Approval record missing: {APPROVAL}")

    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if approval.get("status") != "VISUALLY_APPROVED_BY_USER":
        raise ValueError("The visual approval record is not approved.")

    approved_images = set(approval.get("approved_images", []))
    expected_images = {
        Path(name).with_suffix(".jpg").name for name in APPROVED_FILES
    }
    if not expected_images.issubset(approved_images):
        raise ValueError("Approval record does not include both reviewed images.")

    # Verify candidate files exist before copying anything.
    for name in APPROVED_FILES:
        candidate = CANDIDATES / name
        if not candidate.is_file():
            raise FileNotFoundError(f"Candidate label missing: {candidate}")

    # Copy all source files; never modify the source folder.
    shutil.copytree(SOURCE, DEST)

    replaced = []
    try:
        target_labels = DEST / "train" / "labels"

        for name in APPROVED_FILES:
            candidate = CANDIDATES / name
            target = target_labels / name

            if not target.is_file():
                raise FileNotFoundError(
                    f"Copied training label not found: {target}"
                )

            # Copy candidate labels only into the new dataset.
            shutil.copy2(candidate, target)
            replaced.append(name)

        # Record exactly what changed in the copied dataset.
        manifest = {
            "status": "CREATED_REQUIRES_FINAL_VALIDATION",
            "source_dataset": str(SOURCE.relative_to(ROOT)),
            "corrected_dataset": str(DEST.relative_to(ROOT)),
            "replaced_training_labels": replaced,
            "original_dataset_modified": False,
            "training_started": False,
        }

        (DEST / "correction_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    except Exception:
        # Remove the incomplete copy, never the source.
        shutil.rmtree(DEST, ignore_errors=True)
        raise

    print(json.dumps(manifest, indent=2))
    print("\nSeparate corrected copy created.")
    print("Original dataset was not modified.")
    print("Training was not started.")


if __name__ == "__main__":
    main()
