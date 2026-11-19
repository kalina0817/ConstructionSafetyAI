
import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "reports" / "model_selection" / "comparison_results.json"

st.title("Model Comparison")
st.caption("Saved detector model-selection results")

st.info(
    "These results are loaded from the existing model-selection report. "
    "They are saved evaluation results, not new evaluations."
)

if not RESULTS_FILE.is_file():
    st.error(f"Comparison results file not found: {RESULTS_FILE}")
    st.stop()

try:
    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        results = json.load(file)
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Could not read comparison results: {error}")
    st.stop()

if not isinstance(results, list) or not results:
    st.warning("No model comparison records were found.")
    st.stop()

required_fields = {
    "stage",
    "model",
    "precision",
    "recall",
    "map50",
    "map50_95",
}

valid_results = [
    record
    for record in results
    if isinstance(record, dict)
    and required_fields.issubset(record.keys())
]

if not valid_results:
    st.warning("The comparison file does not contain the expected fields.")
    st.stop()

data = pd.DataFrame(valid_results)

st.subheader("Comparison overview")

st.write(f"**Saved model records:** {len(data)}")
st.write(
    "**Stages:** "
    + ", ".join(sorted(data["stage"].dropna().unique().tolist()))
)

st.divider()

st.subheader("Filter results")

stages = ["All stages"] + sorted(data["stage"].dropna().unique().tolist())
selected_stage = st.selectbox("Detector stage", stages)

if selected_stage == "All stages":
    filtered = data.copy()
else:
    filtered = data[data["stage"] == selected_stage].copy()

display = filtered[
    [
        "stage",
        "model",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]
].copy()

display = display.rename(
    columns={
        "stage": "Stage",
        "model": "Model",
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP50",
        "map50_95": "mAP50-95",
    }
)

metric_columns = ["Precision", "Recall", "mAP50", "mAP50-95"]

st.dataframe(
    display.style.format(
        {column: "{:.4f}" for column in metric_columns}
    ),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader("Model performance")

chart_data = filtered.set_index("model")[
    ["precision", "recall", "map50", "map50_95"]
]

chart_data = chart_data.rename(
    columns={
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP50",
        "map50_95": "mAP50-95",
    }
)

st.bar_chart(chart_data)

st.divider()

st.subheader("Interpretation")

for stage in sorted(filtered["stage"].dropna().unique()):
    stage_data = filtered[filtered["stage"] == stage]

    st.markdown(f"**{stage}**")

    if len(stage_data) == 1:
        st.write(
            "Only one saved candidate is available. "
            "A comparison between alternative models is not possible yet."
        )
        continue

    best = stage_data.loc[stage_data["map50_95"].idxmax()]

    st.write(
        f"Highest saved mAP50-95: **{best['model']}** "
        f"({best['map50_95']:.4f})."
    )

st.caption(
    "Metrics are presented as saved in the comparison file. "
    "Results from different dataset versions or evaluation conditions "
    "should not be treated as directly comparable without checking "
    "their evaluation configurations."
)

# ---------------------------------------------------------
# GPU and CPU Model Benchmark
# ---------------------------------------------------------

st.divider()
st.subheader("GPU and CPU Model Benchmark")

BENCHMARK_FILE = ROOT / "reports" / "model_benchmark" / "combined_model_comparison.json"

if BENCHMARK_FILE.is_file():
    with BENCHMARK_FILE.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    records = benchmark.get("results", [])
    rows = []

    for record in records:
        gpu = record["gpu_validation"]
        cpu = record["local_cpu"]

        rows.append({
            "Dataset": record["dataset"],
            "Model": record["model"],
            "Precision": gpu["precision"],
            "Recall": gpu["recall"],
            "mAP50": gpu["mAP50"],
            "mAP50-95": gpu["mAP50_95"],
            "GPU inference (ms)": gpu["inference_ms"],
            "CPU mean (ms)": cpu["mean_ms_per_image"],
            "CPU median (ms)": cpu["median_ms_per_image"],
            "CPU images/sec": cpu["images_per_second"],
        })

    if rows:
        data = pd.DataFrame(rows)

        selected = st.selectbox(
            "Benchmark dataset",
            ["All datasets"] + sorted(data["Dataset"].unique()),
            key="benchmark_dataset_filter",
        )

        if selected != "All datasets":
            data = data[data["Dataset"] == selected]

        st.subheader("Accuracy and inference results")
        st.dataframe(data, use_container_width=True, hide_index=True)

        st.subheader("CPU inference speed")
        st.bar_chart(data.pivot(
            index="Dataset",
            columns="Model",
            values="CPU mean (ms)",
        ))

        st.subheader("Model recommendations")
        for dataset in sorted(data["Dataset"].unique()):
            st.markdown(f"**{dataset}**")
            st.write(benchmark.get("recommendations", {}).get(dataset, "No recommendation available."))

        st.caption(
            "GPU and CPU timings use different measurement procedures. "
            "Detection metrics do not establish real-world hazard confirmation."
        )
else:
    st.warning(f"Benchmark report not found: {BENCHMARK_FILE}")
