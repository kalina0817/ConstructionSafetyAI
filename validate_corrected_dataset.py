from pathlib import Path
import yaml

dataset = Path("data/processed/construction-ppe-corrected")

print("Dataset exists:", dataset.is_dir())

for split in ["train", "valid", "test"]:
    folder = dataset / split
    if folder.exists():
        images = list((folder / "images").glob("*"))
        labels = list((folder / "labels").glob("*.txt"))
        print(f"{split}: images={len(images)}, labels={len(labels)}")
    else:
        print(f"{split}: folder missing")

yaml_file = dataset / "data.yaml"
print("data.yaml exists:", yaml_file.exists())

if yaml_file.exists():
    print("\nDataset configuration:")
    print(yaml.safe_load(yaml_file.read_text(encoding="utf-8")))
else:
    print("Searching for data.yaml inside the dataset...")
    for file in dataset.rglob("data.yaml"):
        print(file)