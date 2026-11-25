import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unified_inference
from unified_inference import compare_detection_models

def detection(dataset, class_name, box, confidence=0.9):
    return {
        "detector": dataset,
        "class_name": class_name,
        "box_xyxy": box,
        "confidence": confidence,
    }


def test_matching_classes_produce_matching_relationship():
    yolo = [
        detection("PPE", "helmet", [0, 0, 10, 10])
    ]
    rtdetr = [
        detection("PPE", "helmet", [0, 0, 9, 10])
    ]

    results = compare_detection_models(yolo, rtdetr)

    assert len(results) == 1
    assert results[0]["same_semantic_class"] is True
    assert results[0]["relationship"] == "matching_class_overlap"
    assert results[0]["iou"] == 0.9


def test_different_classes_produce_class_disagreement():
    yolo = [
        detection("Vehicles", "Loader", [0, 0, 10, 10])
    ]
    rtdetr = [
        detection("Vehicles", "Roller", [0, 0, 10, 10])
    ]

    results = compare_detection_models(yolo, rtdetr)

    assert len(results) == 1
    assert results[0]["same_semantic_class"] is False
    assert results[0]["relationship"] == "class_disagreement"


def test_one_to_one_matching_prevents_duplicate_matches():
    yolo = [
        detection("Toolset", "wrench", [0, 0, 10, 10])
    ]
    rtdetr = [
        detection("Toolset", "wrench", [0, 0, 9, 10]),
        detection("Toolset", "wrench", [0, 0, 8, 10]),
    ]

    results = compare_detection_models(yolo, rtdetr)

    assert len(results) == 1
    assert results[0]["rtdetr_detection_index"] == 0
    assert results[0]["iou"] == 0.9


def test_different_datasets_are_not_matched():
    yolo = [
        detection("PPE", "helmet", [0, 0, 10, 10])
    ]
    rtdetr = [
        detection("Toolset", "helmet", [0, 0, 10, 10])
    ]

    results = compare_detection_models(yolo, rtdetr)

    assert results == []


def test_detections_below_iou_threshold_are_not_matched():
    yolo = [
        detection("Vehicles", "Excavator", [0, 0, 10, 10])
    ]
    rtdetr = [
        detection("Vehicles", "Excavator", [20, 20, 30, 30])
    ]

    results = compare_detection_models(
        yolo,
        rtdetr,
        iou_threshold=0.50,
    )

    assert results == []
def test_saved_report_comparison_creates_json_report(tmp_path, monkeypatch):
    image_path = str(tmp_path / "test_image.jpg")

    yolo_report = {
        "model_mode": "yolo",
        "input_image": image_path,
        "confidence_threshold": 0.25,
        "image_size": 640,
        "yolo11n_detections": [
            detection("PPE", "helmet", [0, 0, 10, 10])
        ],
    }

    rtdetr_report = {
        "model_mode": "rtdetr",
        "input_image": image_path,
        "confidence_threshold": 0.25,
        "image_size": 640,
        "rtdetr_l_detections": [
            detection("PPE", "helmet", [0, 0, 9, 10]),
            detection("PPE", "vest", [20, 20, 30, 30]),
        ],
    }

    yolo_path = tmp_path / "yolo.json"
    rtdetr_path = tmp_path / "rtdetr.json"
    output_dir = tmp_path / "comparison"

    yolo_path.write_text(json.dumps(yolo_report), encoding="utf-8")
    rtdetr_path.write_text(json.dumps(rtdetr_report), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unified_inference.py",
            "--compare",
            "--yolo-report", str(yolo_path),
            "--rtdetr-report", str(rtdetr_path),
            "--output-dir", str(output_dir),
        ],
    )

    monkeypatch.setattr(
        unified_inference,
        "compare_detection_models",
        unified_inference.compare_detection_models,
    )

    unified_inference.main()

    report_path = output_dir / "model_comparison.json"

    assert report_path.is_file()

    result = json.loads(report_path.read_text(encoding="utf-8"))

    ppe_summary = result["summary_by_dataset"]["PPE"]

    assert ppe_summary["yolo11n_detection_count"] == 1
    assert ppe_summary["rtdetr_l_detection_count"] == 2
    assert ppe_summary["matched_box_count"] == 1
    assert ppe_summary["matching_class_count"] == 1
    assert ppe_summary["class_disagreement_count"] == 0
    assert ppe_summary["yolo_only_detection_count"] == 0
    assert ppe_summary["rtdetr_only_detection_count"] == 1
    assert len(result["comparisons"]) == 1


def test_contained_no_goggle_detections_are_deduplicated():
    detections = [
        detection(
            "PPE",
            "no_goggle",
            [184, 263, 277, 295],
            confidence=0.54831,
        ),
        detection(
            "PPE",
            "no_goggle",
            [184, 252, 279, 352],
            confidence=0.43964,
        ),
    ]

    kept, suppressed, reviews = (
        unified_inference.filter_duplicate_detections(detections)
    )

    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.54831
    assert len(suppressed) == 1
    assert suppressed[0]["candidate"]["confidence"] == 0.43964
    assert len(reviews) == 0


def test_worker_duplicate_does_not_remove_neighbor_or_ppe():
    worker = detection("PPE", "Person", [251, 116, 397, 560], confidence=0.9)
    duplicate = detection("PPE", "Person", [299, 109, 395, 562], confidence=0.8)
    neighbor = detection("PPE", "Person", [385, 138, 479, 563], confidence=0.7)
    vest = detection("PPE", "vest", [300, 180, 390, 370], confidence=0.6)
    kept, suppressed, _ = unified_inference.filter_duplicate_detections(
        [worker, duplicate, neighbor, vest]
    )
    assert kept == [worker, neighbor, vest]
    assert [entry["candidate"] for entry in suppressed] == [duplicate]
