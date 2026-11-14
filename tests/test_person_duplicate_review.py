import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from person_duplicate_review import (
    build_person_overlay,
    load_review_items,
    load_saved_reviews,
    save_review,
)


class PersonDuplicateReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.manifest_path = (
            self.root / "reports" / "person_duplicate_validation_final_40.csv"
        )
        self.run_id = "run_001"
        self.run_dir = self.root / "reports" / "inference_runs" / self.run_id
        self.run_dir.mkdir(parents=True)
        self.image_path = self.root / "sample.jpg"
        Image.new("RGB", (40, 30), "white").save(self.image_path)

        self.report = {
            "input_image": str(self.image_path),
            "raw_person_detection_count": 2,
            "filtered_person_detection_count": 1,
            "possible_duplicate_person_count": 1,
            "possible_duplicate_persons": [{
                "candidate": {
                    "confidence": 0.4,
                    "bbox_xyxy": [5, 5, 20, 25],
                },
                "kept_detection": {
                    "confidence": 0.8,
                    "bbox_xyxy": [4, 4, 22, 28],
                },
                "iou": 0.5,
                "containment": 0.9,
                "area_ratio": 0.6,
                "suppression_reason": "containment_and_area_ratio",
            }],
            "analysis_detections": [{
                "class_name": "Person",
                "confidence": 0.8,
                "bbox_xyxy": [4, 4, 22, 28],
            }],
        }
        (self.run_dir / "detections.json").write_text(
            json.dumps(self.report),
            encoding="utf-8",
        )
        (self.run_dir / "unified_detections.jpg").write_bytes(b"overlay")
        self._write_manifest()

    def _write_manifest(self, rows=None):
        rows = rows or [{
            "image": self.image_path.name,
            "run_id": self.run_id,
            "raw_person_detections": "2",
            "filtered_person_detections": "1",
            "suppressed_person_duplicates": "1",
            "manual_review_status": "Not manually reviewed",
            "manual_review_notes": "No review recorded.",
        }]
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "image",
            "run_id",
            "raw_person_detections",
            "filtered_person_detections",
            "suppressed_person_duplicates",
            "manual_review_status",
            "manual_review_notes",
        ]
        with self.manifest_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_manifest_and_constructs_person_overlay(self):
        items = load_review_items(self.root)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["image"], self.image_path.name)
        overlay = build_person_overlay(items[0])
        self.assertEqual(overlay.size, (40, 30))

    def test_saves_and_reloads_review_without_touching_source_report(self):
        items = load_review_items(self.root)
        review_path = self.root / "reviews.json"
        review = {
            "image": self.image_path.name,
            "run_id": self.run_id,
            "count_assessment": "confident",
            "visible_person_count": 1,
            "suppression_reviews": ["duplicate_of_kept_person"],
            "reviewer_notes": "",
        }

        reviews = save_review(review_path, items, {}, review)
        loaded = load_saved_reviews(review_path, {self.image_path.name})

        self.assertEqual(reviews, loaded)
        self.assertEqual(loaded[self.image_path.name]["visible_person_count"], 1)
        self.assertEqual(
            json.loads(
                (self.run_dir / "detections.json").read_text(
                    encoding="utf-8"
                )
            ),
            self.report,
        )

    def test_rejects_missing_suppressed_detection_decision(self):
        items = load_review_items(self.root)
        with self.assertRaisesRegex(ValueError, "Every suppressed"):
            save_review(
                self.root / "reviews.json",
                items,
                {},
                {
                    "image": self.image_path.name,
                    "count_assessment": "confident",
                    "visible_person_count": 1,
                    "suppression_reviews": [],
                },
            )

    def test_rejects_manifest_count_drift(self):
        row = {
            "image": self.image_path.name,
            "run_id": self.run_id,
            "raw_person_detections": "3",
            "filtered_person_detections": "1",
            "suppressed_person_duplicates": "1",
            "manual_review_status": "Not manually reviewed",
            "manual_review_notes": "",
        }
        self._write_manifest([row])

        with self.assertRaisesRegex(ValueError, "disagree"):
            load_review_items(self.root)


if __name__ == "__main__":
    unittest.main()
