
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

DATASETS = {
    "PPE": ROOT / "data/processed/construction-ppe/data.yaml",
    "Toolset": ROOT / "data/processed/construction-toolset-deduplicated/data.yaml",
    "Vehicles": ROOT / "data/processed/construction-vehicle-detection-clean-loader-corrected/data.yaml",
}

CANDIDATES = {
    "PPE": {
        "ppe_baseline": ROOT / "runs/detect/runs/detect/ppe_yolo11n_baseline/weights/best.pt",
        "ppe_postcorrection_pilot": ROOT / "runs/detect/runs/detect/ppe_yolo11n_postcorrection_pilot/weights/best.pt",
        "ppe_postcorrection_15ep": ROOT / "runs/detect/runs/detect/ppe_yolo11n_postcorrection_15ep/weights/best.pt",
    },
    "Toolset": {
        "toolset35_existing": ROOT / "models/toolset35_final_model/toolset35_yolo11n_best.pt",
    },
    "Vehicles": {
        "vehicle_pilot": ROOT / "runs/detect/runs/detect/vehicle_yolo11n_pilot/weights/best.pt",
        "vehicle_continuation_5ep": ROOT / "runs/detect/runs/detect/vehicle_yolo11n_continuation_5ep/weights/best.pt",
        "vehicle_corrected_dataset_5ep": ROOT / "runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt",
    },
}

OUTPUT_DIR = ROOT / "reports" / "model_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    results_summary = []

    for stage, candidates in CANDIDATES.items():
        dataset = DATASETS[stage]

        if not dataset.exists():
            raise FileNotFoundError(f"Dataset YAML missing: {dataset}")

        print(f"\n{'=' * 65}")
        print(f"STAGE: {stage}")
        print(f"VALIDATION DATASET: {dataset}")
        print(f"{'=' * 65}")

        for name, weights in candidates.items():
            if not weights.exists():
                print(f"\nSKIPPING missing checkpoint: {weights}")
                continue

            print(f"\nEvaluating: {name}")
            print(f"Weights: {weights}")

            model = YOLO(str(weights))

            metrics = model.val(
                data=str(dataset),
                imgsz=640,
                batch=4,
                device="cpu",
                workers=0,
                plots=False,
                project=str(OUTPUT_DIR),
                name=name,
                exist_ok=True,
                verbose=False,
            )

            row = {
                "stage": stage,
                "model": name,
                "weights": str(weights.relative_to(ROOT)),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }

            results_summary.append(row)

            print(f"Precision: {row['precision']:.4f}")
            print(f"Recall:    {row['recall']:.4f}")
            print(f"mAP50:     {row['map50']:.4f}")
            print(f"mAP50-95:  {row['map50_95']:.4f}")

    output_file = OUTPUT_DIR / "comparison_results.json"
    output_file.write_text(
        json.dumps(results_summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nComparison complete.")
    print(f"Results saved to: {output_file}")

    print("\nFINAL SUMMARY")
    for row in results_summary:
        print(
            f"{row['stage']:10} | {row['model']:35} | "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"mAP50={row['map50']:.4f} "
            f"mAP50-95={row['map50_95']:.4f}"
        )


if __name__ == "__main__":
    main()
