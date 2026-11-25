from pathlib import Path
import hashlib
import json
import sys
import traceback

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "capstone"))

from construction_safety.config import MODEL_VARIANTS
from construction_safety.inference import run_inference, save_run

INPUT_DIR = PROJECT / "reports" / "blind_scene_inference_inputs" / "images"
OUTPUT_DIR = PROJECT / "reports" / "blind_scene_inference_20261124"

CHECKPOINTS = {
    "ppe": (
        PROJECT / "runs/detect/runs/detect/ppe_yolo11n_postcorrection_15ep/weights/best.pt",
        "3c6be2472623f8ec daf91a9835e14c10e423313ff9189a64a5263a1e98dd3dbf".replace(" ", ""),
    ),
    "toolset35": (
        PROJECT / "runs/detect/runs/detect/toolset_yolo11n_continuation_final5-2/weights/best.pt",
        "a9c4caee0f585a8f4a90215f23429d20eef36341df87e20e220d1190a75a6f47",
    ),
    "vehicles": (
        PROJECT / "runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt",
        "4f769e836f282973f381af9687e132dcfc1b9423615a1a656af2c4e7b56e13db",
    ),
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if OUTPUT_DIR.exists():
    raise SystemExit(f"STOP: Output folder already exists: {OUTPUT_DIR}")

images = sorted(INPUT_DIR.glob("IMG_*.jpg"))
if len(images) != 18:
    raise SystemExit(f"Expected 18 images, found {len(images)}")

print("Verifying checkpoints...")
for name, (path, expected_hash) in CHECKPOINTS.items():
    if not path.is_file():
        raise SystemExit(f"Missing checkpoint: {path}")

    actual_hash = sha256(path)
    if actual_hash.lower() != expected_hash.lower():
        raise SystemExit(f"Hash mismatch for {name}: {actual_hash}")

    print(f"VERIFIED {name}: {actual_hash}")

# Use the existing verified Toolset35 checkpoint for the
# toolset35 detector, only in this Python process.
MODEL_VARIANTS["toolset35"]["yolo11n"]["checkpoint"] = (
    MODEL_VARIANTS["toolset"]["yolo11n"]["checkpoint"]
)

OUTPUT_DIR.mkdir(parents=True)
results = []

print("\nStarting application inference on 18 images using CPU.")

for index, image in enumerate(images, start=1):
    print(f"\n[{index}/18] Processing {image.name}", flush=True)

    try:
        annotated_image, report = run_inference(
            image_path=image,
            confidence=0.25,
            image_size=640,
            model_names=("ppe", "toolset35", "vehicles"),
        )

        # Confirm the report records the exact intended checkpoints.
        for name, (_, expected_hash) in CHECKPOINTS.items():
            recorded = report["checkpoint_provenance"][name]["sha256"]
            if recorded.lower() != expected_hash.lower():
                raise RuntimeError(
                    f"{name} report hash mismatch: {recorded}"
                )

        run_folder, report_path = save_run(
            report,
            output_root=OUTPUT_DIR / image.stem,
        )

        annotated_image.save(run_folder / "annotated.jpg")

        results.append({
            "image": image.name,
            "status": "SUCCESS",
            "report": str(report_path.relative_to(PROJECT)),
            "detection_summary": report["detection_summary"],
            "screening_outcome": report["screening_outcome"],
        })

        print(f"Saved report: {report_path}", flush=True)

    except Exception as error:
        results.append({
            "image": image.name,
            "status": "FAILED",
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        print(f"FAILED: {error}", flush=True)

summary_path = OUTPUT_DIR / "batch_summary.json"
summary_path.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

successes = sum(row["status"] == "SUCCESS" for row in results)
failures = len(results) - successes

print(f"\nCompleted: {successes} successful, {failures} failed.")
print(f"Summary: {summary_path}")
