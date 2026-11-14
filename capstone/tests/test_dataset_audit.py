import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from construction_safety.dataset_audit import audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        for split in ("train", "valid", "test"):
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "labels").mkdir(parents=True)
            image_path = self.root / split / "images" / "scene.jpg"
            Image.new("RGB", (20, 20), "white").save(image_path)
            (self.root / split / "labels" / "scene.txt").write_text(
                "0 0.5 0.5 0.4 0.4\n",
                encoding="utf-8",
            )
        self.data_yaml = self.root / "data.yaml"
        self.data_yaml.write_text(
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n"
            "nc: 1\n"
            "names: [Person]\n",
            encoding="utf-8",
        )

    def test_audits_split_pairs_and_class_counts(self):
        report = audit_dataset(self.data_yaml)

        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["splits"]["train"]["image_count"], 1)
        self.assertEqual(report["splits"]["val"]["annotation_count"], 1)
        self.assertEqual(
            report["splits"]["test"]["annotations_by_class"]["Person"],
            1,
        )

    def test_missing_label_is_reported_without_assuming_negative(self):
        (self.root / "valid" / "labels" / "scene.txt").unlink()

        report = audit_dataset(self.data_yaml)

        self.assertEqual(
            report["splits"]["val"]["images_without_label_file"],
            1,
        )
        self.assertIn(
            "cannot determine which",
            report["interpretation_note"],
        )

    def test_invalid_box_is_reported(self):
        (self.root / "test" / "labels" / "scene.txt").write_text(
            "0 0.5 0.5 2 0.4\n",
            encoding="utf-8",
        )

        report = audit_dataset(self.data_yaml)

        self.assertEqual(report["error_count"], 1)
        self.assertIn("normalized", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
