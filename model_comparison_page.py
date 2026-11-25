import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
BENCHMARK_FILE = (
    ROOT / "reports" / "model_benchmark" / "combined_model_comparison.json"
)

st.title("Compare results")
st.write("Two questions guide the study. How well does the app find objects? What does it say about possible risks around workers?")
risk, objects = st.tabs(["Checking for risks", "Finding objects"])

with risk:
    st.subheader("The 12-photo study")
    st.write("All three methods start with the same objects found in each photo. We compare what they say about workers near machines.")
    study = ROOT / "research" / "worker_machine_blind_12" / "evaluation"
    for column, label in zip(st.columns(3), ["Scene-graph rules", "Qwen3-VL", "MiniCPM-V"]):
        count = 0
        for number in range(1, 13):
            photo = f"Photo_{number:02d}"
            if label == "Scene-graph rules":
                path = study / "yolo11n_rules_baseline" / photo / "detections.json"
            else:
                path = study / "visual_comparison" / label / photo / "assessment.json"
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(saved, dict) and (saved.get("reasoning_status") == "COMPLETED" or
                    (label == "Scene-graph rules" and "safety_assessment" in saved)):
                count += 1
        column.metric(label, f"{count} / 12", border=True)
    st.caption("Counts show finished photo checks. The study still needs an independent review of the answers.")
    comparison_folder = study / "visual_comparison"
    report_path = comparison_folder / "Morning report.md"
    if report_path.is_file():
        report_text = report_path.read_text(encoding="utf-8")
        with st.expander("Read the findings from all three methods"):
            st.markdown(report_text.partition("\n")[2].split("## Photo-by-photo results")[0])
        st.download_button(
            "Download the full 12-photo report",
            data=report_text,
            file_name="Construction Safety AI - 12-photo study.md",
            mime="text/markdown",
        )
    st.divider()

with objects:
    if not BENCHMARK_FILE.is_file():
        st.error(f"Benchmark results file not found: {BENCHMARK_FILE}")
        st.stop()

    try:
        with BENCHMARK_FILE.open("r", encoding="utf-8") as file:
            benchmark = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read benchmark results: {error}")
        st.stop()

    records = benchmark.get("results", [])

    if not isinstance(records, list) or not records:
        st.warning("No model comparison results were found.")
        st.stop()


    def model_label(value):
        name = str(value).lower().replace("-", "").replace("_", "")
        if "rtdetr" in name:
            return "RT-DETR-L"
        if "yolo" in name:
            return "YOLO11n"
        return str(value)


    def dataset_label(value):
        name = str(value).lower().replace("_", "").replace("-", "")
        labels = {
            "ppe": "Protective gear",
            "toolset35": "Tools",
            "vehicles": "Vehicles",
        }
        return labels.get(name, str(value))


    rows = []

    for record in records:
        try:
            gpu = record["gpu_validation"]
            cpu = record["local_cpu"]

            rows.append({
                "Dataset": dataset_label(record["dataset"]),
                "Model": model_label(record["model"]),
                "Precision": float(gpu["precision"]),
                "Recall": float(gpu["recall"]),
                "mAP50": float(gpu["mAP50"]),
                "mAP50-95": float(gpu["mAP50_95"]),
                "GPU inference (ms)": float(gpu["inference_ms"]),
                "CPU mean (ms)": float(cpu["mean_ms_per_image"]),
                "CPU median (ms)": float(cpu["median_ms_per_image"]),
                "CPU images/sec": float(cpu["images_per_second"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        st.error("The benchmark file is missing expected result fields.")
        st.stop()

    data = pd.DataFrame(rows)

    st.header("Test overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Photo groups tested", data["Dataset"].nunique())
    col2.metric("AI models compared", data["Model"].nunique())
    col3.metric("Saved test results", len(data))

    st.caption(
        "The saved tests measured how often the app found the right object. Speed was measured "
        "separately on a graphics card and on this computer."
    )

    with st.expander("How to read the scores"):
        st.write("One score counts how often a label was right. Another counts how many of the known objects the app found. "
                 "The two location scores also check how closely the boxes line up with the objects. Higher scores are better.")
    st.divider()
    st.header("Choose a photo group")

    datasets = ["All photo groups"] + sorted(data["Dataset"].unique().tolist())
    selected = st.selectbox("Photo group", datasets)

    filtered = (
        data.copy()
        if selected == "All photo groups"
        else data[data["Dataset"] == selected].copy()
    )

    st.divider()
    st.header("Compare the two models")

    for dataset in sorted(filtered["Dataset"].unique()):
        subset = filtered[filtered["Dataset"] == dataset].copy()

        st.subheader(dataset)

        if subset["Model"].nunique() < 2:
            st.warning("Only one model has saved results for this photo group.")
            st.dataframe(subset, width="stretch", hide_index=True)
            continue

        subset = subset.drop_duplicates(subset=["Model"], keep="last")
        subset = subset.set_index("Model")

        accuracy = subset[
            ["Precision", "Recall", "mAP50", "mAP50-95"]
        ].T * 100

        accuracy.index = [
            "Correct labels (%)",
            "Known objects found (%)",
            "Score for finding and locating objects (%)",
            "Score with stricter box checks (%)",
        ]

        st.write("**How well did each model find objects?**")
        st.dataframe(
            accuracy.style.format("{:.2f}%"),
            width="stretch",
        )

        speed = subset[
            ["GPU inference (ms)", "CPU mean (ms)", "CPU images/sec"]
        ].T

        speed.index = [
            "Time with a graphics card (milliseconds)",
            "Time on this computer (milliseconds)",
            "Photos per second on this computer",
        ]

        st.write("**How quickly did each model analyze images?**")
        st.dataframe(
            speed.style.format("{:.2f}"),
            width="stretch",
        )

        best_map = subset["mAP50-95"].idxmax()
        best_recall = subset["Recall"].idxmax()
        fastest_cpu = subset["CPU mean (ms)"].idxmin()

        st.write("**What do these results tell us?**")
        st.write(
            f"- Best result across stricter object-location checks: "
            f"**{best_map}** "
            f"({subset.loc[best_map, 'mAP50-95'] * 100:.2f}%)."
        )
        st.write(
            f"- Found the largest share of actual objects: **{best_recall}** "
            f"({subset.loc[best_recall, 'Recall'] * 100:.2f}%)."
        )
        st.write(
            f"- Fastest on this computer: **{fastest_cpu}** "
            f"({subset.loc[fastest_cpu, 'CPU mean (ms)']:.2f} milliseconds per image)."
        )

        st.divider()

    st.header("Results across photo groups")

    accuracy_data = filtered[
        ["Dataset", "Model", "Precision", "Recall", "mAP50", "mAP50-95"]
    ].copy()

    for metric in ["Precision", "Recall", "mAP50", "mAP50-95"]:
        accuracy_data[metric] *= 100

    for metric, heading in [
        ("mAP50", "Finding and locating objects (%)"),
        ("mAP50-95", "Object-location accuracy under stricter checks (%)"),
        ("Recall", "Share of actual objects found (%)"),
        ("Precision", "Share of correct AI predictions (%)"),
    ]:
        st.write(f"**{heading}**")
        chart = accuracy_data.pivot(
            index="Dataset",
            columns="Model",
            values=metric,
        )
        st.bar_chart(chart)

    st.divider()
    st.header("Speed on this computer")

    st.write(
        "These results show how long each model takes to analyze an image "
        "on this computer. Shorter bars mean faster analysis."
    )

    cpu_chart = filtered.pivot(
        index="Dataset",
        columns="Model",
        values="CPU mean (ms)",
    )
    st.bar_chart(cpu_chart)

    st.header("Speed with a graphics card")

    st.write(
        "These results show how long each model takes to analyze an image "
        "with the graphics card used for testing. Shorter bars mean faster analysis."
    )

    gpu_chart = filtered.pivot(
        index="Dataset",
        columns="Model",
        values="GPU inference (ms)",
    )
    st.bar_chart(gpu_chart)

    st.divider()
    st.header("What do the overall results show?")

    st.markdown(
        """
        - **YOLO11n** checked photos faster on this computer.
        - **RT-DETR-L** scored higher on protective gear and vehicles in these tests.
        - **YOLO11n** found a larger share of the tools and scored higher on their locations.
        """
    )

    recommendations = benchmark.get("recommendations", {})

    if recommendations:
        st.subheader("Notes from the original tests")

        for dataset, recommendation in recommendations.items():
            st.markdown(f"**{dataset_label(dataset)}**")
            st.write(recommendation)

with risk:
    st.header("Earlier checks on the same photo")

    st.write(
        """
        These saved checks used both MiniCPM-V and Qwen3-VL on the same photo with the same instructions.
        The results show where their answers matched and how long each took.
        """
    )

    runs_dir = ROOT / "reports" / "inference_runs"
    reasoning_rows = []

    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            mini_path = run_dir / "vision_language_minicpm.json"
            qwen_path = run_dir / "vision_language_qwen.json"

            if not (mini_path.is_file() and qwen_path.is_file()):
                continue

            try:
                with mini_path.open("r", encoding="utf-8") as file:
                    mini = json.load(file)
                with qwen_path.open("r", encoding="utf-8") as file:
                    qwen = json.load(file)

                # Keep only completed, matched image-and-prompt pairs.
                if (
                    mini.get("status") != "COMPLETED"
                    or qwen.get("status") != "COMPLETED"
                    or mini.get("image_path") != qwen.get("image_path")
                    or mini.get("prompt_version") != qwen.get("prompt_version")
                ):
                    continue

                mini_analysis = mini.get("analysis") or {}
                qwen_analysis = qwen.get("analysis") or {}

                mini_ppe = mini_analysis.get("potential_missing_ppe") or {}
                qwen_ppe = qwen_analysis.get("potential_missing_ppe") or {}

                mini_interaction = (
                    mini_analysis.get("potential_worker_machinery_interaction") or {}
                )
                qwen_interaction = (
                    qwen_analysis.get("potential_worker_machinery_interaction") or {}
                )

                mini_metadata = mini.get("metadata") or {}
                qwen_metadata = qwen.get("metadata") or {}

                image_path = mini.get("image_path") or ""

                reasoning_rows.append({
                    "Run": run_dir.name,
                    "Image": Path(image_path).name if image_path else "Unknown",
                    "Set of instructions": mini.get("prompt_version", "Unknown"),
                    "MiniCPM gear check": mini_ppe.get("decision", "unknown"),
                    "Qwen gear check": qwen_ppe.get("decision", "unknown"),
                    "MiniCPM interaction": mini_interaction.get(
                        "decision", "unknown"
                    ),
                    "Qwen interaction": qwen_interaction.get(
                        "decision", "unknown"
                    ),
                    "MiniCPM time (sec)": mini_metadata.get("elapsed_seconds"),
                    "Qwen time (sec)": qwen_metadata.get("elapsed_seconds"),
                    "MiniCPM thinking disabled": mini_metadata.get(
                        "thinking_disabled"
                    ),
                    "Qwen thinking disabled": qwen_metadata.get(
                        "thinking_disabled"
                    ),
                })

            except (OSError, json.JSONDecodeError, TypeError, AttributeError):
                # Ignore incomplete or malformed run files without stopping the page.
                continue

    if reasoning_rows:
        reasoning_df = pd.DataFrame(reasoning_rows)

        st.metric("Completed matched image pairs", len(reasoning_df))

        st.subheader("Answers for each photo")
        st.dataframe(
            reasoning_df,
            width="stretch",
            hide_index=True,
        )

        st.subheader("How often did the answers match?")

        ppe_agreement = int(
            (reasoning_df["MiniCPM gear check"] == reasoning_df["Qwen gear check"]).sum()
        )
        interaction_agreement = int(
            (
                reasoning_df["MiniCPM interaction"]
                == reasoning_df["Qwen interaction"]
            ).sum()
        )

        col1, col2 = st.columns(2)
        col1.metric(
            "Matching answers about protective gear",
            f"{ppe_agreement}/{len(reasoning_df)}",
        )
        col2.metric(
            "Matching answers about workers near machines",
            f"{interaction_agreement}/{len(reasoning_df)}",
        )

        st.subheader("How often each answer appeared")

        for heading, mini_column, qwen_column in [
            (
                "Gear that may be missing",
                "MiniCPM gear check",
                "Qwen gear check",
            ),
            (
                "Potential worker-machinery interaction",
                "MiniCPM interaction",
                "Qwen interaction",
            ),
        ]:
            st.write(f"**{heading}**")
            distribution = pd.concat(
                [
                    reasoning_df[mini_column].value_counts().rename("MiniCPM-V"),
                    reasoning_df[qwen_column].value_counts().rename("Qwen3-VL"),
                ],
                axis=1,
            ).fillna(0).astype(int)
            st.bar_chart(distribution)

        st.subheader("Reasoning time")

        timing = reasoning_df[
            ["MiniCPM time (sec)", "Qwen time (sec)"]
        ].apply(pd.to_numeric, errors="coerce")

        timing_summary = pd.DataFrame(
            {
                "Average seconds": [
                    timing["MiniCPM time (sec)"].mean(),
                    timing["Qwen time (sec)"].mean(),
                ],
                "Median seconds": [
                    timing["MiniCPM time (sec)"].median(),
                    timing["Qwen time (sec)"].median(),
                ],
            },
            index=["MiniCPM-V", "Qwen3-VL"],
        )

        st.dataframe(
            timing_summary.style.format("{:.2f}"),
            width="stretch",
        )
        st.bar_chart(timing_summary[["Average seconds"]])

        st.subheader("Comparison limitations")
        st.warning(
            """
            Two methods can agree and still miss a hazard. These earlier checks show what each method said.
            Different settings can also affect how long a check takes.
            """
        )
    else:
        st.info(
            "No earlier checks with both methods using the same photo "
            "and instructions were found."
        )


    st.caption("Check the photo and site conditions before acting on a finding.")
