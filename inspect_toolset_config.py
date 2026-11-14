
from pathlib import Path
import yaml

paths = [
    Path(r"data\raw\construction-toolset\data.yaml"),
    Path(r"data\processed\construction-toolset-deduplicated\data.yaml"),
]

for path in paths:
    print(f"\n--- {path} ---")

    if not path.exists():
        print("File not found:", path.resolve())
        continue

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, value in config.items():
            print(f"{key}: {value}")
    except Exception as error:
        print("Could not read configuration:", error)

print("\nInspection complete. No files were modified.")
