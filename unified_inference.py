
from pathlib import Path
from datetime import datetime
import argparse
import json

import cv2
from ultralytics import YOLO, RTDETR

from screening_assessment import assess_worker_ppe
from reasoning_engine import (
    MACHINERY_CLASSES,
    PPE_OBJECT_CLASSES,
    TOOLSET_TOOL_CLASSES,
    build_scene_graph,
    evaluate_safety_rules,
    generate_review_candidates,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
FORBIDDEN_OBSERVATION_PHRASES = (
    "human-confirmed",
    "human confirmed",
    "confirmed missing ",
    "confirmed absence of ",
    "confirmed lack of ",
    "confirmed ppe violation",
    "confirmed ppe finding",
)


def validate_automatic_findings(report):
    """Reject automatic findings that incorrectly imply human confirmation."""

    assessment = report.get("safety_assessment", {})
    findings = assessment.get("findings", [])

    if not isinstance(findings, list):
        raise ValueError(
            "Invalid safety assessment: findings must be a list."
        )

    issues = []

    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(
                f"Invalid safety finding at index {index}."
            )

        observation = finding.get("observation", "")

        if not isinstance(observation, str):
            continue

        observation_lower = observation.lower()

        for phrase in FORBIDDEN_OBSERVATION_PHRASES:
            if phrase in observation_lower:
                finding_id = finding.get(
                    "finding_id", f"index {index}"
                )
                issues.append(
                    f"{finding_id}: {observation}"
                )
                break

    if issues:
        raise ValueError(
            "Automatic finding validation failed:\n"
            + "\n".join(issues)
        )

    return len(findings)

MODEL_PATHS = {
    "PPE": ROOT / "runs/detect/runs/detect/ppe_yolo11n_postcorrection_15ep/weights/best.pt",
    "Toolset": ROOT / "models/toolset35_final_model/toolset35_yolo11n_best.pt",
    "Vehicles": ROOT / "runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt",
}

MODEL_COLORS = {
    "PPE": (0, 200, 0),
    "Toolset": (255, 150, 0),
    "Vehicles": (0, 100, 255),
}
RTDETR_MODEL_PATHS = {
    "PPE": ROOT / "models/rtdetr_l_comparison/ConstructionSafetyAI_model_comparison/rtdetr_l_ppe_best.pt",
    "Toolset": ROOT / "models/rtdetr_l_comparison/ConstructionSafetyAI_model_comparison/rtdetr_l_toolset35_best.pt",
    "Vehicles": ROOT / "models/rtdetr_l_comparison/ConstructionSafetyAI_model_comparison/rtdetr_l_vehicles_best.pt",
}

# ============================================================
# BOUNDING BOX UTILITIES
# ============================================================

def calculate_iou(box_a, box_b):
    """Calculate intersection over union for XYXY bounding boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# SEMANTIC CLASS MAPPING
# ============================================================

def get_semantic_class(class_name):
    """Map approved equivalent labels to a shared object category."""

    aliases = {
        # PPE and Toolset equivalents
        "helmet": "helmet",
        "safety helmet": "helmet",
        "gloves": "gloves",
        "vest": "vest",
        "safety_vest": "vest",
        "goggles": "goggles",
        "safety glass": "goggles",

        # Toolset and Vehicles equivalent
        "excavator": "excavator",

        # Toolset equivalent labels
        "shovel": "shovel",
        "spade": "shovel",
        "knife": "knife",
        "utility knife": "knife",
        "measuring tape": "measuring tool",
        "meter": "measuring tool",
        "ruler": "measuring tool",
    }

    normalized_name = class_name.strip().lower()
    return aliases.get(normalized_name, normalized_name)


# ============================================================
# GLOBAL DUPLICATE DETECTION FILTER
# ============================================================

def filter_duplicate_detections(
    detections,
    iou_threshold=0.60,
    containment_threshold=0.85,
    min_area_ratio=0.30,
):
    """
    Identify likely duplicate physical objects across all detectors.

    Same semantic class:
        Suppress overlapping duplicate detections using IoU or
        containment and area ratio.

    Different semantic classes:
        Suppress only nearly identical boxes. Other overlaps are
        recorded for review instead of automatically removed.

    The highest-confidence detection is retained.
    Original detections are not modified.
    """

    ordered_detections = sorted(
        detections,
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    kept = []
    suppressed = []
    overlap_reviews = []

    for candidate in ordered_detections:

        candidate_box = candidate["box_xyxy"]
        candidate_class = get_semantic_class(
            candidate["class_name"]
        )

        duplicate_of = None
        duplicate_metrics = None

        for existing in kept:

            existing_box = existing["box_xyxy"]
            existing_class = get_semantic_class(
                existing["class_name"]
            )

            # A person and another class are separate objects.
            if (candidate_class == "person") != (existing_class == "person"):
                continue

            iou = calculate_iou(
                candidate_box,
                existing_box,
            )

            x_left = max(candidate_box[0], existing_box[0])
            y_top = max(candidate_box[1], existing_box[1])
            x_right = min(candidate_box[2], existing_box[2])
            y_bottom = min(candidate_box[3], existing_box[3])

            intersection_area = (
                max(0, x_right - x_left)
                * max(0, y_bottom - y_top)
            )

            candidate_area = (
                max(0, candidate_box[2] - candidate_box[0])
                * max(0, candidate_box[3] - candidate_box[1])
            )

            existing_area = (
                max(0, existing_box[2] - existing_box[0])
                * max(0, existing_box[3] - existing_box[1])
            )

            smaller_area = min(candidate_area, existing_area)
            larger_area = max(candidate_area, existing_area)

            containment = (
                intersection_area / smaller_area
                if smaller_area > 0 else 0.0
            )

            area_ratio = (
                smaller_area / larger_area
                if larger_area > 0 else 0.0
            )

            same_semantic_class = (
                candidate_class == existing_class
            )

            if same_semantic_class:
                # Use the person-specific area floor from the 40-image validation.
                area_floor = max(min_area_ratio, 0.35) if candidate_class == "person" else min_area_ratio
                is_duplicate = (
                    iou >= iou_threshold
                    or (
                        containment >= containment_threshold
                        and area_ratio >= area_floor
                    )
                )
            else:
                # Different semantic classes are never automatically
                # suppressed based on bounding-box overlap.
                is_duplicate = False

                if iou >= iou_threshold:
                    overlap_reviews.append({
                        "candidate": candidate,
                        "overlapping_detection": existing,
                        "candidate_class": candidate["class_name"],
                        "overlapping_class": existing["class_name"],
                        "iou": round(iou, 4),
                        "status": "different_class_overlap_requires_review",
                    })

            if is_duplicate:
                duplicate_of = existing
                duplicate_metrics = {
                    "iou": round(iou, 4),
                    "containment": round(containment, 4),
                    "area_ratio": round(area_ratio, 4),
                    "same_semantic_class": same_semantic_class,
                }
                break

        if duplicate_of is None:
            kept.append(candidate)
        else:
            suppressed.append({
                "candidate": candidate,
                "kept_detection": duplicate_of,
                **duplicate_metrics,
                "reason": "duplicate_object_overlap",
            })

    return kept, suppressed, overlap_reviews


# ============================================================
# CROSS-MODEL COMPARISON
# ============================================================

def compare_cross_model_detections(
    detections,
    iou_threshold=0.50,
):
    """Find overlapping detections produced by different models."""

    comparisons = []

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):

            first = detections[i]
            second = detections[j]

            if first["detector"] == second["detector"]:
                continue

            iou = calculate_iou(
                first["box_xyxy"],
                second["box_xyxy"],
            )

            if iou >= iou_threshold:

                same_semantic_class = (
                    get_semantic_class(first["class_name"])
                    == get_semantic_class(second["class_name"])
                )

                relationship = (
                    "possible_same_object"
                    if same_semantic_class
                    else "overlap_requires_review"
                )

                comparisons.append({
                    "detection_index_a": i,
                    "detection_index_b": j,
                    "source_detection_index_a": first.get(
                        "source_detection_index", i
                    ),
                    "source_detection_index_b": second.get(
                        "source_detection_index", j
                    ),
                    "detector_a": first["detector"],
                    "class_a": first["class_name"],
                    "confidence_a": first["confidence"],
                    "detector_b": second["detector"],
                    "class_b": second["class_name"],
                    "confidence_b": second["confidence"],
                    "iou": round(iou, 4),
                    "relationship": relationship,
                    "status": "potential_overlap_review",
                })

    return comparisons

# ============================================================
# YOLO11n VS RT-DETR-L SAME-DATASET COMPARISON
# ============================================================

def compare_detection_models(
    yolo_detections,
    rtdetr_detections,
    iou_threshold=0.50,
):
    """
    Compare YOLO11n and RT-DETR-L detections within each dataset.

    Uses greedy one-to-one matching, selecting the highest-IoU
    eligible pairs first. This comparison does not affect safety
    reasoning or merge predictions.
    """

    candidates = []

    for yolo_index, yolo in enumerate(yolo_detections):
        for rtdetr_index, rtdetr in enumerate(rtdetr_detections):

            if yolo["detector"] != rtdetr["detector"]:
                continue

            iou = calculate_iou(
                yolo["box_xyxy"],
                rtdetr["box_xyxy"],
            )

            if iou >= iou_threshold:
                candidates.append(
                    (iou, yolo_index, rtdetr_index, yolo, rtdetr)
                )

    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_yolo = set()
    matched_rtdetr = set()
    comparisons = []

    for iou, yolo_index, rtdetr_index, yolo, rtdetr in candidates:

        if yolo_index in matched_yolo:
            continue

        if rtdetr_index in matched_rtdetr:
            continue

        matched_yolo.add(yolo_index)
        matched_rtdetr.add(rtdetr_index)

        same_class = (
            get_semantic_class(yolo["class_name"])
            == get_semantic_class(rtdetr["class_name"])
        )

        comparisons.append({
            "dataset": yolo["detector"],
            "yolo_detection_index": yolo_index,
            "rtdetr_detection_index": rtdetr_index,
            "yolo_class": yolo["class_name"],
            "rtdetr_class": rtdetr["class_name"],
            "yolo_confidence": yolo["confidence"],
            "rtdetr_confidence": rtdetr["confidence"],
            "iou": round(iou, 4),
            "same_semantic_class": same_class,
            "relationship": (
                "matching_class_overlap"
                if same_class
                else "class_disagreement"
            ),
        })

    return comparisons


# ============================================================
# SAVED REPORT COMPARISON
# ============================================================

# Comparison-only mode reuses compare_detection_models above.
# It does not load detector checkpoints or run safety reasoning.


# ============================================================
# PERSON-MACHINERY SPATIAL ANALYSIS
# ============================================================

def analyze_person_machinery_proximity(
    detections,
    image_width,
    image_height,
):
    """
    Measure image-space distance between detected people
    and detected vehicles or heavy machinery.

    This does not establish real-world physical distance or danger.
    """

    people = [
        (index, detection)
        for index, detection in enumerate(detections)
        if detection["class_name"].strip().lower() == "person"
    ]

    machinery_classes = {
        "boringmachine",
        "bulldozer",
        "dump truck",
        "excavator",
        "forklift",
        "grader",
        "loader",
        "mixer truck",
        "mobile crane",
        "roller",
    }

    machinery = [
        (index, detection)
        for index, detection in enumerate(detections)
        if get_semantic_class(detection["class_name"])
        in machinery_classes
    ]

    image_diagonal = (
        image_width ** 2 + image_height ** 2
    ) ** 0.5

    relationships = []

    for person_index, person in people:

        px1, py1, px2, py2 = person["box_xyxy"]

        person_center_x = (px1 + px2) / 2
        person_center_y = (py1 + py2) / 2

        for machine_index, machine in machinery:

            mx1, my1, mx2, my2 = machine["box_xyxy"]

            machine_center_x = (mx1 + mx2) / 2
            machine_center_y = (my1 + my2) / 2

            dx = person_center_x - machine_center_x
            dy = person_center_y - machine_center_y

            pixel_distance = (dx ** 2 + dy ** 2) ** 0.5

            normalized_distance = (
                pixel_distance / image_diagonal
                if image_diagonal > 0 else 0.0
            )

            relationships.append({
                "person_analysis_index": person_index,
                "person_source_detection_index": person.get(
                    "source_detection_index", person_index
                ),
                "person_class": person["class_name"],
                "person_detector": person["detector"],
                "person_confidence": person["confidence"],
                "person_box_xyxy": person["box_xyxy"],
                "machinery_analysis_index": machine_index,
                "machinery_source_detection_index": machine.get(
                    "source_detection_index", machine_index
                ),
                "machinery_class": machine["class_name"],
                "machinery_detector": machine["detector"],
                "machinery_confidence": machine["confidence"],
                "machinery_box_xyxy": machine["box_xyxy"],
                "center_distance_pixels": round(pixel_distance, 2),
                "normalized_image_distance": round(
                    normalized_distance, 5
                ),
                "status": "potential_spatial_relationship_review",
            })

    relationships.sort(
        key=lambda item: item["normalized_image_distance"]
    )

    return {
        "person_count": len(people),
        "machinery_count": len(machinery),
        "relationship_count": len(relationships),
        "relationships": relationships,
        "interpretation_note": (
            "Distances are measured in image space between bounding-box "
            "centers. They do not establish physical distance, actual "
            "contact, unsafe proximity, or a confirmed hazard."
        ),
    }


# ============================================================
# SCENE INVENTORY
# ============================================================

def build_scene_inventory(detections):
    """Organize independent detector predictions into scene categories."""

    groups = {
        "people": [],
        "ppe_predictions": [],
        "tools_and_equipment": [],
        "vehicles_and_heavy_machinery": [],
        "unclassified": [],
    }

    ppe_names = {
        name.replace("_", " ").lower()
        for name in PPE_OBJECT_CLASSES
    }

    tool_names = {
        name.replace("_", " ").lower()
        for name in TOOLSET_TOOL_CLASSES
    }

    machinery_names = {
        name.replace("_", " ").lower()
        for name in MACHINERY_CLASSES
    }

    for detection in detections:

        normalized_class = (
            detection["class_name"].strip().lower().replace("_", " ")
        )

        if normalized_class == "person":
            category = "people"

        elif normalized_class.startswith("no "):
            category = "ppe_predictions"

        elif normalized_class in ppe_names:
            category = "ppe_predictions"

        elif normalized_class in tool_names:
            category = "tools_and_equipment"

        elif normalized_class in machinery_names:
            category = "vehicles_and_heavy_machinery"

        else:
            category = "unclassified"

        groups[category].append(detection.copy())

    counts = {
        category: len(items)
        for category, items in groups.items()
    }

    return {
        "total_detections": len(detections),
        "category_counts": counts,
        "groups": groups,
        "interpretation_note": (
            "Categories organize detector predictions only. "
            "They do not establish object identity, PPE compliance, "
            "or the presence of a physical hazard."
        ),
    }


# ============================================================
# MAIN INFERENCE PIPELINE
# ============================================================

def classify_screening_outcome(findings, detector_names):
    """Classify the overall visual screening result."""

    findings = findings or []
    selected = set(detector_names)

    if findings:
        outcome = "POTENTIAL"
        explanation = (
            f"The screening rules generated {len(findings)} "
            "potential finding(s). These require interpretation "
            "and do not confirm a hazard."
        )
    elif {"PPE", "Vehicles"}.issubset(selected):
        outcome = "NO_SUPPORTED"
        explanation = (
            "No findings were generated by the currently implemented "
            "screening rules. This does not establish that the site is safe."
        )
    else:
        outcome = "INSUFFICIENT"
        explanation = (
            "The selected detector configuration does not cover all "
            "currently implemented screening rules."
        )

    return {
        "outcome": outcome,
        "explanation": explanation,
        "finding_count": len(findings),
        "detector_coverage": {
            "PPE": "PPE" in selected,
            "Toolset": "Toolset" in selected,
            "Vehicles": "Vehicles" in selected,
        },
        "high_confidence_alert_available": False,
        "high_confidence_alert_limitation": (
            "No independently validated hazard-specific criterion "
            "currently supports a high-confidence alert."
        ),
        "interpretation_limit": (
            "NO_SUPPORTED does not mean safe. INSUFFICIENT means "
            "the selected detector configuration does not cover "
            "all current screening rules."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run ConstructionSafetyAI YOLO11n, RT-DETR-L, or both."
    )

    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--compare", action="store_true",
                        help="Compare two saved inference reports without loading models.")
    parser.add_argument("--yolo-report", type=Path,
                        help="Saved YOLO11n detections.json report.")
    parser.add_argument("--rtdetr-report", type=Path,
                        help="Saved RT-DETR-L detections.json report.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["yolo", "rtdetr", "both"],
        default="yolo",
        help="Run YOLO11n or RT-DETR-L separately. Use --compare for saved reports.",
    )

    parser.add_argument(
        "--ppe-model",
        choices=["yolo", "rtdetr"],
        default=None,
        help="Architecture for the PPE detector.",
    )
    parser.add_argument(
        "--toolset-model",
        choices=["yolo", "rtdetr"],
        default=None,
        help="Architecture for the Toolset-35 detector.",
    )
    parser.add_argument(
        "--vehicles-model",
        choices=["yolo", "rtdetr"],
        default=None,
        help="Architecture for the Vehicles detector.",
    )

    parser.add_argument(
        "--screening-model", choices=["yolo", "rtdetr"],
        help="Use one architecture for all screening checkpoints and build a scene graph.",
    )
    parser.add_argument("--graph-only", action="store_true",
                        help="Build screening evidence for external reasoning.")
    args = parser.parse_args()
    if args.graph_only and not args.screening_model:
        parser.error("--graph-only requires --screening-model.")

    if args.screening_model:
        if args.compare or any((args.ppe_model, args.toolset_model, args.vehicles_model)):
            parser.error("--screening-model cannot be combined with per-task models or --compare.")
        args.ppe_model = args.toolset_model = args.vehicles_model = args.screening_model


    independent_mode = any(
        value is not None
        for value in (
            args.ppe_model,
            args.toolset_model,
            args.vehicles_model,
        )
    )

    model_selection = {
        "PPE": args.ppe_model or "yolo",
        "Toolset": args.toolset_model or "yolo",
        "Vehicles": args.vehicles_model or "yolo",
    }

    if args.compare:
        if args.image:
            parser.error("--image cannot be used with --compare.")
        if not args.yolo_report or not args.rtdetr_report:
            parser.error("--compare requires both --yolo-report and --rtdetr-report.")

        yolo_path = args.yolo_report.resolve()
        rtdetr_path = args.rtdetr_report.resolve()

        if not yolo_path.is_file():
            parser.error(f"YOLO report not found: {yolo_path}")
        if not rtdetr_path.is_file():
            parser.error(f"RT-DETR report not found: {rtdetr_path}")

        with yolo_path.open("r", encoding="utf-8") as file:
            yolo_report = json.load(file)
        with rtdetr_path.open("r", encoding="utf-8") as file:
            rtdetr_report = json.load(file)

        if yolo_report.get("model_mode") != "yolo":
            parser.error("The YOLO report must come from a --model-mode yolo run.")
        if rtdetr_report.get("model_mode") != "rtdetr":
            parser.error("The RT-DETR report must come from a --model-mode rtdetr run.")

        if Path(yolo_report.get("input_image", "")).resolve() != Path(
            rtdetr_report.get("input_image", "")
        ).resolve():
            parser.error("Reports use different input images.")

        if yolo_report.get("confidence_threshold") != rtdetr_report.get("confidence_threshold"):
            parser.error("Reports use different confidence thresholds.")
        if yolo_report.get("image_size") != rtdetr_report.get("image_size"):
            parser.error("Reports use different image sizes.")

        yolo_detections = yolo_report.get("yolo11n_detections")
        rtdetr_detections = rtdetr_report.get("rtdetr_l_detections")

        if not isinstance(yolo_detections, list) or not isinstance(rtdetr_detections, list):
            parser.error("A report is missing its model detection list.")

        comparisons = compare_detection_models(
            yolo_detections,
            rtdetr_detections,
            iou_threshold=0.50,
        )

        summary = {}
        for dataset in ("PPE", "Toolset", "Vehicles"):
            yolo_items = [
                item for item in yolo_detections
                if item.get("detector") == dataset
            ]
            rtdetr_items = [
                item for item in rtdetr_detections
                if item.get("detector") == dataset
            ]
            dataset_matches = [
                item for item in comparisons
                if item["dataset"] == dataset
            ]

            matched_yolo = {
                item["yolo_detection_index"] for item in dataset_matches
            }
            matched_rtdetr = {
                item["rtdetr_detection_index"] for item in dataset_matches
            }

            yolo_indices = {
                index for index, item in enumerate(yolo_detections)
                if item.get("detector") == dataset
            }
            rtdetr_indices = {
                index for index, item in enumerate(rtdetr_detections)
                if item.get("detector") == dataset
            }

            summary[dataset] = {
                "yolo11n_detection_count": len(yolo_items),
                "rtdetr_l_detection_count": len(rtdetr_items),
                "matched_box_count": len(dataset_matches),
                "matching_class_count": sum(
                    item["same_semantic_class"] for item in dataset_matches
                ),
                "class_disagreement_count": sum(
                    not item["same_semantic_class"] for item in dataset_matches
                ),
                "yolo_only_detection_count": len(yolo_indices - matched_yolo),
                "rtdetr_only_detection_count": len(rtdetr_indices - matched_rtdetr),
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else ROOT / "reports" / "model_comparisons" / timestamp
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        comparison_report = {
            "input_image": yolo_report["input_image"],
            "confidence_threshold": yolo_report["confidence_threshold"],
            "image_size": yolo_report["image_size"],
            "models": ["YOLO11n", "RT-DETR-L"],
            "comparison_method": "Greedy one-to-one bounding-box IoU matching with semantic class comparison.",
            "iou_threshold": 0.50,
            "source_yolo_report": str(yolo_path),
            "source_rtdetr_report": str(rtdetr_path),
            "summary_by_dataset": summary,
            "comparisons": comparisons,
            "interpretation_note": (
                "These are model prediction comparisons, not ground truth. "
                "Unmatched detections are not automatically false positives or false negatives. "
                "Predictions are not merged and are not used for safety reasoning."
            ),
        }

        output_file = output_dir / "model_comparison.json"
        output_file.write_text(
            json.dumps(comparison_report, indent=4),
            encoding="utf-8",
        )

        print("\nSaved-report comparison completed.")
        print(f"Input image: {comparison_report['input_image']}")
        print("\nComparison summary:")
        for dataset, values in summary.items():
            print(f"\n{dataset}:")
            for label, value in values.items():
                print(f"  {label}: {value}")
        print(f"\nComparison report: {output_file}")
        return

    if args.yolo_report or args.rtdetr_report:
        parser.error("--yolo-report and --rtdetr-report require --compare.")
    if not args.image:
        parser.error("--image is required unless --compare is used.")

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive.")

    image_path = Path(args.image).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if independent_mode:
        for name, architecture in model_selection.items():
            paths = MODEL_PATHS if architecture == "yolo" else RTDETR_MODEL_PATHS
            checkpoint = paths[name]
            if not checkpoint.is_file():
                model_label = "YOLO11n" if architecture == "yolo" else "RT-DETR-L"
                parser.error(f"{name} {model_label} checkpoint not found: {checkpoint}")
    else:
        if args.model_mode in ("yolo", "both"):
            for name, path in MODEL_PATHS.items():
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{name} YOLO11n checkpoint not found: {path}"
                    )

        if args.model_mode in ("rtdetr", "both"):
            for name, path in RTDETR_MODEL_PATHS.items():
                if not path.is_file():
                    raise FileNotFoundError(
                        f"RT-DETR-L {name} checkpoint not found: {path}"
                    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    annotated = image.copy()
    yolo_detections = []
    rtdetr_detections = []

    # --------------------------------------------------------
    # Inference: independent per-dataset selection or legacy mode
    # --------------------------------------------------------

    independent_selected_detections = []

    if independent_mode:
        for dataset_name, architecture in model_selection.items():
            if architecture == "yolo":
                model_path = MODEL_PATHS[dataset_name]
                model = YOLO(str(model_path))
                model_label = "YOLO11n"
                destination = yolo_detections
            else:
                model_path = RTDETR_MODEL_PATHS[dataset_name]
                model = RTDETR(str(model_path))
                model_label = "RT-DETR-L"
                destination = rtdetr_detections

            print(f"\\nLoading {model_label} {dataset_name} model...")

            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device="cpu",
                verbose=False,
            )

            result = results[0]
            dataset_detections = []

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detection = {
                    "detector": dataset_name,
                    "model_variant": model_label,
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": round(confidence, 5),
                    "box_xyxy": [x1, y1, x2, y2],
                }
                destination.append(detection)
                dataset_detections.append(detection)

            independent_selected_detections.extend(dataset_detections)
            print(
                f"{model_label} {dataset_name}: "
                f"{len(result.boxes)} detections"
            )

    else:
        # Original YOLO11n inference mode.
        if args.model_mode in ("yolo", "both"):
            for model_name, model_path in MODEL_PATHS.items():
                print(f"\\nLoading YOLO11n {model_name} model...")
                model = YOLO(str(model_path))

                results = model.predict(
                    source=str(image_path),
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device="cpu",
                    verbose=False,
                )

                result = results[0]

                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    yolo_detections.append({
                        "detector": model_name,
                        "model_variant": "YOLO11n",
                        "class_id": class_id,
                        "class_name": model.names[class_id],
                        "confidence": round(confidence, 5),
                        "box_xyxy": [x1, y1, x2, y2],
                    })

                print(
                    f"YOLO11n {model_name}: {len(result.boxes)} detections"
                )

        # Original RT-DETR-L inference mode.
        if args.model_mode in ("rtdetr", "both"):
            for model_name, model_path in RTDETR_MODEL_PATHS.items():
                print(f"\\nLoading RT-DETR-L {model_name} model...")
                model = RTDETR(str(model_path))

                results = model.predict(
                    source=str(image_path),
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device="cpu",
                    verbose=False,
                )

                result = results[0]

                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    rtdetr_detections.append({
                        "detector": model_name,
                        "model_variant": "RT-DETR-L",
                        "class_id": class_id,
                        "class_name": model.names[class_id],
                        "confidence": round(confidence, 5),
                        "box_xyxy": [x1, y1, x2, y2],
                    })

                print(
                    f"RT-DETR-L {model_name}: {len(result.boxes)} detections"
                )

    # Preserve source indices separately for each model.
    for index, detection in enumerate(yolo_detections):
        detection["source_detection_index"] = index

    for index, detection in enumerate(rtdetr_detections):
        detection["source_detection_index"] = index

    # --------------------------------------------------------
    # Model comparisons
    # --------------------------------------------------------

    model_comparisons = []
    model_comparison_summary = {}

    if args.model_mode == "both":
        model_comparisons = compare_detection_models(
            yolo_detections,
            rtdetr_detections,
            iou_threshold=0.50,
        )

        for dataset_name in MODEL_PATHS:
            dataset_comparisons = [
                item for item in model_comparisons
                if item["dataset"] == dataset_name
            ]

            model_comparison_summary[dataset_name] = {
                "yolo11n_raw_detection_count": sum(
                    1 for item in yolo_detections
                    if item["detector"] == dataset_name
                ),
                "rtdetr_l_raw_detection_count": sum(
                    1 for item in rtdetr_detections
                    if item["detector"] == dataset_name
                ),
                "overlap_comparison_count": len(dataset_comparisons),
                "matching_class_overlap_count": sum(
                    1 for item in dataset_comparisons
                    if item["same_semantic_class"]
                ),
                "class_disagreement_count": sum(
                    1 for item in dataset_comparisons
                    if not item["same_semantic_class"]
                ),
            }

    # Use one selected architecture per dataset in independent mode.
    if independent_mode:
        selected_detections = independent_selected_detections
        for index, detection in enumerate(selected_detections):
            detection["source_detection_index"] = index
    elif args.model_mode == "rtdetr":
        selected_detections = rtdetr_detections
    else:
        selected_detections = yolo_detections

    cross_model_comparisons = compare_cross_model_detections(
        selected_detections,
        iou_threshold=0.50,
    )

    cross_model_class_disagreements = [
        item for item in cross_model_comparisons
        if item.get("relationship") == "overlap_requires_review"
    ]

    # --------------------------------------------------------
    # Duplicate filtering and scene inventory
    # --------------------------------------------------------

    (
        analysis_detections,
        suppressed_duplicate_detections,
        possible_cross_class_duplicates,
    ) = filter_duplicate_detections(
        selected_detections,
        iou_threshold=0.60,
        containment_threshold=0.85,
        min_area_ratio=0.30,
    )

    scene_inventory = build_scene_inventory(analysis_detections)

    # --------------------------------------------------------
    # Draw detections
    # --------------------------------------------------------

    display_detections = analysis_detections

    for index, detection in enumerate(display_detections):
        x1, y1, x2, y2 = detection["box_xyxy"]
        model_name = detection["detector"]
        color = MODEL_COLORS.get(model_name, (255, 255, 255))

        label = (
            f"{detection['model_variant']} {model_name}: "
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        if args.screening_model and detection["class_name"].strip().lower() == "person":
            label = f"object_{index:04d} {label}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Safety reasoning
    # RT-DETR-only mode does not run YOLO safety rules.
    # --------------------------------------------------------

    if args.model_mode == "rtdetr" and not independent_mode:
        reasoning_detections = []
        scene_graph = {
            "status": "NOT_RUN",
            "reason": "YOLO-based safety reasoning is not run in RT-DETR-only mode.",
        }
        spatial_analysis = {
            "person_count": 0,
            "machinery_count": 0,
            "relationship_count": 0,
            "relationships": [],
            "status": "NOT_RUN",
            "interpretation_note": (
                "YOLO-based spatial analysis was not run in RT-DETR-only mode."
            ),
        }
        safety_assessment = {
            "status": "NOT_RUN",
            "findings": [],
            "message": (
                "RT-DETR-L detections are displayed, but YOLO-based "
                "safety reasoning was not run."
            ),
        }
        screening_outcome = {
            "outcome": "INSUFFICIENT",
            "explanation": (
                "RT-DETR-L detection was run without the implemented "
                "YOLO-based safety reasoning."
            ),
            "finding_count": 0,
        }
        review_analysis = {
            "candidate_count": 0,
            "candidates": [],
            "status": "NOT_RUN",
        }

    else:
        reasoning_detections = analysis_detections

        spatial_analysis = analyze_person_machinery_proximity(
            reasoning_detections,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )

        scene_graph = build_scene_graph(
            reasoning_detections,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )

        if args.graph_only:
            safety_assessment = {"findings": [], "status": "AWAITING_REASONING"}
            screening_outcome = {
                "outcome": "INSUFFICIENT",
                "explanation": "The selected reasoning model has yet to assess the scene.",
            }
        else:
            worker_ppe = assess_worker_ppe(scene_graph) if args.screening_model else None
            safety_assessment = evaluate_safety_rules(scene_graph, worker_ppe=worker_ppe)
            screening_outcome = classify_screening_outcome(
                safety_assessment.get("findings", []),
                MODEL_PATHS.keys(),
            )
            if worker_ppe is not None and not safety_assessment["findings"] and (
                not worker_ppe or any(row["status"] == "unassessable" for row in worker_ppe)
            ):
                screening_outcome.update(
                    outcome="INSUFFICIENT",
                    explanation="The rules found no supported alert, but worker PPE coverage was incomplete.",
                    interpretation_limit="The image does not establish PPE status for all five checks per worker.",
                )

        review_analysis = generate_review_candidates(
            scene_graph,
            cross_model_comparisons=cross_model_comparisons,
        )

    # --------------------------------------------------------
    # Analysis counts
    # --------------------------------------------------------

    raw_person_count = sum(
        1 for detection in selected_detections
        if detection["class_name"].strip().lower() == "person"
    )

    filtered_person_count = sum(
        1 for detection in analysis_detections
        if detection["class_name"].strip().lower() == "person"
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nScene inventory:")
    for category, count in scene_inventory["category_counts"].items():
        print(f"{category}: {count}")

    print("\nDetection summary:")
    print(
        "Selected model mode: "
        + (args.screening_model or ("independent per-dataset selection" if independent_mode else args.model_mode))
    )
    if independent_mode:
        print(f"Selected architectures: {model_selection}")
    print(f"Raw detections: {len(selected_detections)}")
    print(f"Retained detections: {len(analysis_detections)}")
    print(
        "Suppressed duplicates: "
        f"{len(suppressed_duplicate_detections)}"
    )
    print(
        "Different-class overlaps requiring review: "
        f"{len(possible_cross_class_duplicates)}"
    )

    print("\nPerson detection counts:")
    print(f"Original person detections: {raw_person_count}")
    print(f"Retained person detections: {filtered_person_count}")

    print("\nScreening outcome:")
    print(screening_outcome)

    print("\nOptional review analysis:")
    print(f"Review candidates: {review_analysis['candidate_count']}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "reports" / "inference_runs" / timestamp
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    image_output = output_dir / "unified_detections.jpg"
    json_output = output_dir / "detections.json"

    if not cv2.imwrite(str(image_output), annotated):
        raise IOError(
            f"Could not save annotated image: {image_output}"
        )

    report = {
        "input_image": str(image_path),
        "confidence_threshold": args.conf,
        "image_size": args.imgsz,
        "device": "cpu",
        "model_mode": args.screening_model or ("independent" if independent_mode else args.model_mode),
        "screening_checkpoints": {
            name: str((MODEL_PATHS if architecture == "yolo" else RTDETR_MODEL_PATHS)[name])
            for name, architecture in model_selection.items()
        } if args.screening_model else None,
        "model_selection": (
            {
                name: "YOLO11n" if architecture == "yolo" else "RT-DETR-L"
                for name, architecture in model_selection.items()
            }
            if independent_mode else None
        ),
        "total_detections": len(selected_detections),
        "retained_detection_count": len(analysis_detections),
        "suppressed_duplicate_count": len(
            suppressed_duplicate_detections
        ),
        "different_class_overlap_review_count": len(
            possible_cross_class_duplicates
        ),
        "raw_person_detection_count": raw_person_count,
        "filtered_person_detection_count": filtered_person_count,
        "duplicate_filter_thresholds": {
            "iou": 0.60,
            "containment": 0.85,
            "minimum_area_ratio": 0.30,
        },
        "suppressed_duplicate_detections": (
            suppressed_duplicate_detections
        ),
        "possible_cross_class_duplicates": (
            possible_cross_class_duplicates
        ),
        "detections": selected_detections,
        "selected_model_detections": selected_detections,
        "yolo11n_detections": yolo_detections,
        "rtdetr_l_detections": rtdetr_detections,
        "analysis_detections": analysis_detections,
        "reasoning_detections": reasoning_detections,
        "cross_model_comparisons": cross_model_comparisons,
        "cross_model_class_disagreements": (
            cross_model_class_disagreements
        ),
        "model_comparison": {
            "models": ["YOLO11n", "RT-DETR-L"],
            "comparison_method": (
                "Same-dataset semantic class comparison "
                "and bounding-box IoU."
            ),
            "iou_threshold": 0.50,
            "summary_by_dataset": model_comparison_summary,
            "comparisons": model_comparisons,
            "interpretation_note": (
                "YOLO11n and RT-DETR-L predictions are preserved "
                "separately. Their predictions are not merged for "
                "scene analysis or safety reasoning."
            ),
        },
        "scene_inventory": scene_inventory,
        "spatial_analysis": spatial_analysis,
        "scene_graph": scene_graph,
        "safety_assessment": safety_assessment,
        "screening_outcome": screening_outcome,
        "review_analysis": review_analysis,
        "interpretation_note": (
            "Detector outputs are predictions, not confirmed observations. "
            "Bounding-box overlap does not prove physical object identity. "
            "Image-space proximity does not establish real-world distance "
            "or danger. Missing PPE detections do not prove PPE absence. "
            "NO_SUPPORTED does not mean the site is safe. This system is "
            "a screening aid, not a safety certification."
        ),
    }

    validated_finding_count = validate_automatic_findings(report)

    print(
        "\nAutomatic finding validation passed: "
        f"{validated_finding_count} finding(s)."
    )

    json_output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nUnified inference completed.")
    print(f"Total selected detections: {len(selected_detections)}")
    print(f"Retained detections: {len(analysis_detections)}")
    print(f"Annotated image: {image_output}")
    print(f"Detection report: {json_output}")


# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()




