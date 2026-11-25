"""Refresh the public demo from the completed 12-photo study."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from screening_assessment import assess_worker_ppe

def read(path):
    return json.loads(path.read_bytes())

def main():
    study = ROOT / "research/worker_machine_blind_12"
    target = ROOT / "docs"
    pictures = target / "photos"
    pictures.mkdir(parents=True, exist_ok=True)
    photos = {}
    for number in range(1, 13):
        photo = f"Photo_{number:02d}"
        baseline = study / "evaluation/yolo11n_rules_baseline" / photo
        detection = read(baseline / "detections.json")
        assert set(detection["model_selection"].values()) == {"YOLO11n"}
        graph = detection["scene_graph"]
        results = {}
        for method in ["Scene-graph rules", "MiniCPM-V", "Qwen3-VL"]:
            if method == "Scene-graph rules":
                report = detection
                gear = assess_worker_ppe(graph)
            else:
                report = read(study / "evaluation/visual_comparison" / method / photo / "assessment.json")
                assert report["reasoning_model"] == method and report["reasoning_status"] == "COMPLETED"
                gear = report["worker_ppe"]
            results[method] = {"outcome": report["screening_outcome"], "findings": report["safety_assessment"]["findings"],
                               "gear": gear, "relationships": graph["relationships"],
                               "machine_checks": report.get("worker_machinery_assessments", [])}
        photos[photo] = {"objects": graph["nodes"], "methods": results}
        shutil.copyfile(study / "reviewer/photos" / f"{photo}.jpg", pictures / f"{photo}.jpg")
        shutil.copyfile(baseline / "unified_detections.jpg", pictures / f"{photo}-marked.jpg")
    (target / "results.json").write_text(json.dumps(photos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

if __name__ == "__main__":
    main()
