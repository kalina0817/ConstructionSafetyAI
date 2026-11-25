import pandas as pd

path = "reports/model_benchmarks/per_class_test_comparison.csv"
df = pd.read_csv(path)

metrics = ["precision", "recall", "map50", "map50_95"]

# Macro averages across classes with ground truth
valid = df[df["ground_truth_present"] == True]

print("\n=== PER-CLASS MACRO AVERAGES ===")
print(
    valid.groupby(["dataset", "model"])[metrics]
    .mean()
    .round(4)
    .to_string()
)

print("\n=== FIVE WEAKEST CLASSES BY mAP50 ===")
for dataset in ["ppe", "toolset", "vehicles"]:
    print(f"\n{dataset.upper()}")
    for model in ["YOLO11n", "RT-DETR-L"]:
        subset = valid[
            (valid["dataset"] == dataset) &
            (valid["model"] == model)
        ]
        print(f"\n{model}")
        print(
            subset.nsmallest(5, "map50")[
                ["class_name", "precision", "recall", "map50", "map50_95"]
            ].round(4).to_string(index=False)
        )

print("\n=== MISSING-PPE CLASS RECALL ===")
ppe = valid[
    (valid["dataset"] == "ppe") &
    (valid["class_name"].str.startswith("no_"))
]
print(
    ppe[["model", "class_name", "precision", "recall", "map50"]]
    .sort_values(["class_name", "model"])
    .round(4)
    .to_string(index=False)
)
