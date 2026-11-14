
from pathlib import Path
from datetime import datetime
import argparse
import json

import cv2
from ultralytics import YOLO

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

MODEL_PATHS = {
    "PPE": ROOT / "runs/detect/runs/detect/ppe_yolo11n_postcorrection_15ep/weights/best.pt",
    "Toolset": ROOT / "models/toolset35_final_model/toolset35_yolo11n_best.pt",
    "Vehicles": ROOT / "runs/detect/runs/detect/vehicle_yolo11n_continuation_5ep/weights/best.pt",
}

MODEL_COLORS = {
    "PPE": (0, 200, 0),
    "Toolset": (255, 150, 0),
    "Vehicles": (0, 100, 255),
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
# DUPLICATE PERSON DETECTION FILTER
# ============================================================

def filter_duplicate_person_detections(
    detections,
    iou_threshold=0.60,
    containment_threshold=0.85,
    min_area_ratio=0.35
):
    """
    Identify likely duplicate person detections from the same detector.

    Duplicate conditions:

    1. IoU >= iou_threshold

    OR

    2. Containment >= containment_threshold AND
       smaller/larger bounding-box area >= min_area_ratio.

    The highest-confidence detection is retained.

    Original detections are not modified.

    This is an analytical filtering step, not proof of
    real-world object identity.
    """

    person_detections = [
        detection for detection in detections
        if detection["class_name"].strip().lower() == "person"
    ]

    person_detections.sort(
        key=lambda detection: detection["confidence"],
        reverse=True
    )

    kept = []
    suppressed = []

    for candidate in person_detections:

        duplicate_of = None
        duplicate_iou = 0.0
        duplicate_containment = 0.0
        duplicate_area_ratio = 0.0

        candidate_box = candidate["bbox_xyxy"]

        for existing in kept:

            # Only compare detections from the same detector.
            if candidate["detector"] != existing["detector"]:
                continue

            existing_box = existing["bbox_xyxy"]

            iou = calculate_iou(candidate_box, existing_box)

            # Calculate intersection.
            x_left = max(candidate_box[0], existing_box[0])
            y_top = max(candidate_box[1], existing_box[1])
            x_right = min(candidate_box[2], existing_box[2])
            y_bottom = min(candidate_box[3], existing_box[3])

            intersection_width = max(0, x_right - x_left)
            intersection_height = max(0, y_bottom - y_top)

            intersection_area = (
                intersection_width * intersection_height
            )

            # Calculate bounding-box areas.
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

            # Containment measures overlap relative to smaller box.
            containment = (
                intersection_area / smaller_area
                if smaller_area > 0 else 0.0
            )

            # Prevent a much larger box from suppressing a smaller
            # detection based on containment alone.
            area_ratio = (
                smaller_area / larger_area
                if larger_area > 0 else 0.0
            )

            is_duplicate = (
                iou >= iou_threshold
                or (
                    containment >= containment_threshold
                    and area_ratio >= min_area_ratio
                )
            )

            if is_duplicate:

                duplicate_of = existing
                duplicate_iou = iou
                duplicate_containment = containment
                duplicate_area_ratio = area_ratio

                break

        if duplicate_of is None:

            kept.append(candidate)

        else:

            suppression_reason = (
                "iou"
                if duplicate_iou >= iou_threshold
                else "containment_and_area_ratio"
            )

            suppressed.append({
                "candidate": candidate,
                "kept_detection": duplicate_of,
                "iou": round(duplicate_iou, 4),
                "containment": round(duplicate_containment, 4),
                "area_ratio": round(duplicate_area_ratio, 4),
                "suppression_reason": suppression_reason
            })

    return kept, suppressed


# ============================================================
# SEMANTIC CLASS MAPPING
# ============================================================

def get_semantic_class(class_name):
    """Map known class-name aliases to a shared semantic category."""

    aliases = {
        "helmet": "helmet",
        "safety helmet": "helmet",
    }

    normalized_name = class_name.strip().lower()

    return aliases.get(normalized_name, normalized_name)


# ============================================================
# CROSS-MODEL COMPARISON
# ============================================================

def compare_cross_model_detections(
    detections,
    iou_threshold=0.50
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
                first["bbox_xyxy"],
                second["bbox_xyxy"]
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
                    "source_detection_index_a": first.get("source_detection_index", i),
                    "source_detection_index_b": second.get("source_detection_index", j),
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


def suppress_duplicate_excavator_detections(
    detections,
    comparisons,
    iou_threshold=0.60
):
    """Suppress lower-confidence cross-model duplicate excavator detections."""

    candidates = []

    for comparison in comparisons:
        if comparison.get("relationship") != "possible_same_object":
            continue

        if comparison.get("iou", 0) < iou_threshold:
            continue

        if comparison.get("class_a", "").strip().lower() != "excavator":
            continue

        if comparison.get("class_b", "").strip().lower() != "excavator":
            continue

        detector_pair = {
            comparison.get("detector_a"),
            comparison.get("detector_b"),
        }

        if detector_pair != {"Toolset", "Vehicles"}:
            continue

        candidates.append(comparison)

    candidates.sort(key=lambda item: item["iou"], reverse=True)

    matched_indices = set()
    suppressed_indices = set()
    review_records = []

    for comparison in candidates:
        index_a = comparison["source_detection_index_a"]
        index_b = comparison["source_detection_index_b"]

        if index_a in matched_indices or index_b in matched_indices:
            continue

        detection_a = detections[index_a]
        detection_b = detections[index_b]

        confidence_a = detection_a["confidence"]
        confidence_b = detection_b["confidence"]

        matched_indices.update({index_a, index_b})

        if confidence_a == confidence_b:
            review_records.append({
                **comparison,
                "decision": "retained_confidence_tie",
                "reason": "Equal confidence; both detections retained for review.",
            })
            continue

        if confidence_a > confidence_b:
            kept_index, suppressed_index = index_a, index_b
        else:
            kept_index, suppressed_index = index_b, index_a

        suppressed_indices.add(suppressed_index)

        review_records.append({
            **comparison,
            "decision": "suppressed_lower_confidence",
            "kept_source_detection_index": kept_index,
            "suppressed_source_detection_index": suppressed_index,
            "reason": "Higher-confidence cross-model excavator detection retained.",
        })

    return suppressed_indices, review_records


# ============================================================
# PERSON-MACHINERY SPATIAL ANALYSIS
# ============================================================

def analyze_person_machinery_proximity(
    detections,
    image_width,
    image_height
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

        px1, py1, px2, py2 = person["bbox_xyxy"]

        person_center_x = (px1 + px2) / 2
        person_center_y = (py1 + py2) / 2

        for machine_index, machine in machinery:

            mx1, my1, mx2, my2 = machine["bbox_xyxy"]

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
                "person_bbox_xyxy": person["bbox_xyxy"],
                "machinery_analysis_index": machine_index,
                "machinery_source_detection_index": machine.get(
                    "source_detection_index", machine_index
                ),
                "machinery_class": machine["class_name"],
                "machinery_detector": machine["detector"],
                "machinery_confidence": machine["confidence"],
                "machinery_bbox_xyxy": machine["bbox_xyxy"],
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

    for detection in detections:
        normalized_class = (
            detection["class_name"].strip().lower().replace("_", " ")
        )

        if normalized_class == "person":
            category = "people"
        elif normalized_class.startswith("no "):
            category = "ppe_predictions"
        elif normalized_class in {
            name.replace("_", " ") for name in PPE_OBJECT_CLASSES
        }:
            category = "ppe_predictions"
        elif normalized_class in {
            name.replace("_", " ") for name in TOOLSET_TOOL_CLASSES
        }:
            category = "tools_and_equipment"
        elif normalized_class in {
            name.replace("_", " ") for name in MACHINERY_CLASSES
        }:
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

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the three ConstructionSafetyAI detectors on one image."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit output directory for this inference run."
    )

    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")
    if args.imgsz <= 0:
        parser.error("--imgsz must be a positive integer.")

    image_path = Path(args.image).resolve()

    # --------------------------------------------------------
    # Validate input image and model checkpoints.
    # --------------------------------------------------------

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    for name, path in MODEL_PATHS.items():

        if not path.is_file():
            raise FileNotFoundError(
                f"{name} checkpoint not found: {path}"
            )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    annotated = image.copy()

    all_detections = []

    # --------------------------------------------------------
    # Run all three detectors.
    # --------------------------------------------------------

    for model_name, model_path in MODEL_PATHS.items():

        print(f"\nLoading {model_name} model...")

        model = YOLO(str(model_path))

        print(f"Running {model_name} detection...")

        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            device="cpu",
            verbose=False,
        )

        result = results[0]

        color = MODEL_COLORS[model_name]

        for box in result.boxes:

            class_id = int(box.cls[0].item())

            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            class_name = model.names[class_id]

            label = (
                f"{model_name}: {class_name} {confidence:.2f}"
            )

            detection = {
                "detector": model_name,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 5),
                "bbox_xyxy": [x1, y1, x2, y2],
            }

            all_detections.append(detection)

            # Draw original detector predictions.
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
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

        print(
            f"{model_name}: {len(result.boxes)} detections"
        )

    # Preserve original detection indices across filtering.
    for source_index, detection in enumerate(all_detections):
        detection["source_detection_index"] = source_index

    # --------------------------------------------------------
    # Cross-model comparison.
    # --------------------------------------------------------

    cross_model_comparisons = compare_cross_model_detections(
        all_detections,
        iou_threshold=0.50
    )

    # --------------------------------------------------------
    # Duplicate person detection filtering.
    # --------------------------------------------------------

    filtered_person_detections, possible_duplicate_persons = (
        filter_duplicate_person_detections(
            all_detections,
            iou_threshold=0.60,
            containment_threshold=0.85,
            min_area_ratio=0.35
        )
    )

    # --------------------------------------------------------
    # Build analysis detections.
    # Keep all non-person detections and filtered persons.
    # --------------------------------------------------------

    analysis_detections = [
        detection for detection in all_detections
        if detection["class_name"].strip().lower() != "person"
    ] + filtered_person_detections

    # --------------------------------------------------------
    # Scene inventory.
    # --------------------------------------------------------

    scene_inventory = build_scene_inventory(
        analysis_detections
    )

    # --------------------------------------------------------
    # Suppress duplicate Toolset/Vehicles excavator detections.
    # --------------------------------------------------------

    suppressed_excavator_indices, possible_duplicate_excavators = (
        suppress_duplicate_excavator_detections(
            all_detections,
            cross_model_comparisons,
            iou_threshold=0.60,
        )
    )

    reasoning_detections = [
        detection for detection in analysis_detections
        if detection.get("source_detection_index")
        not in suppressed_excavator_indices
    ]

    # --------------------------------------------------------
    # Spatial analysis.
    # --------------------------------------------------------

    spatial_analysis = analyze_person_machinery_proximity(
        reasoning_detections,
        image_width=image.shape[1],
        image_height=image.shape[0]
    )

    # --------------------------------------------------------
    # Review candidates.
    # --------------------------------------------------------

    scene_graph = build_scene_graph(
        reasoning_detections,
        image_width=image.shape[1],
        image_height=image.shape[0],
    )
    safety_assessment = evaluate_safety_rules(scene_graph)
    review_analysis = generate_review_candidates(
        scene_graph
    )

    # --------------------------------------------------------
    # Print analysis results.
    # --------------------------------------------------------

    print("\nScene inventory:")

    for category, count in scene_inventory[
        "category_counts"
    ].items():

        print(f"{category}: {count}")

    print("\nPerson detection filtering:")

    raw_person_count = sum(
        1 for detection in all_detections
        if detection["class_name"].strip().lower() == "person"
    )

    print(f"Raw person detections: {raw_person_count}")

    print(
        f"Filtered person detections: "
        f"{len(filtered_person_detections)}"
    )

    print(
        f"Possible duplicate persons suppressed: "
        f"{len(possible_duplicate_persons)}"
    )

    print("\nPerson-machinery spatial analysis:")

    print(
        f"People: {spatial_analysis['person_count']}"
    )

    print(
        f"Machinery: {spatial_analysis['machinery_count']}"
    )

    print(
        f"Relationships: {spatial_analysis['relationship_count']}"
    )

    print("\nRule-based review analysis:")

    print(
        f"Review candidates: {review_analysis['candidate_count']}"
    )

    print(
        f"Potential cross-model overlaps: "
        f"{len(cross_model_comparisons)}"
    )

    # --------------------------------------------------------
    # Save results in a new timestamped folder.
    # --------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "reports" / "inference_runs" / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    image_output = output_dir / "unified_detections.jpg"

    json_output = output_dir / "detections.json"

    # --------------------------------------------------------
    # Save annotated image.
    # --------------------------------------------------------

    if not cv2.imwrite(
        str(image_output),
        annotated
    ):
        raise IOError(
            f"Could not save annotated image: {image_output}"
        )

    # --------------------------------------------------------
    # Create JSON report.
    # --------------------------------------------------------

    report = {
        "input_image": str(image_path),

        "confidence_threshold": args.conf,

        "image_size": args.imgsz,

        "device": "cpu",

        "total_detections": len(all_detections),

        "raw_person_detection_count": raw_person_count,

        "filtered_person_detection_count": len(
            filtered_person_detections
        ),

        "possible_duplicate_person_count": len(
            possible_duplicate_persons
        ),

        "duplicate_filter_thresholds": {
            "iou": 0.60,
            "containment": 0.85,
            "minimum_area_ratio": 0.35,
        },

        "possible_duplicate_persons": possible_duplicate_persons,
        "possible_duplicate_excavator_count": len(
            suppressed_excavator_indices
        ),
        "duplicate_excavator_iou_threshold": 0.60,
        "possible_duplicate_excavators": possible_duplicate_excavators,
        "reasoning_detections": reasoning_detections,

        "detections": all_detections,

        "analysis_detections": analysis_detections,

        "cross_model_comparisons": cross_model_comparisons,

        "scene_inventory": scene_inventory,

        "spatial_analysis": spatial_analysis,

        "scene_graph": scene_graph,

        "safety_assessment": safety_assessment,

        "review_analysis": review_analysis,

        "interpretation_note": (
            "Independent detector outputs. Overlapping boxes are possible "
            "duplicates requiring review. Semantic matches are candidates, "
            "not confirmed object identities. Detections alone do not "
            "establish PPE compliance or prove a physical hazard."
        ),
    }

    json_output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Completion summary.
    # --------------------------------------------------------

    print("\nUnified inference completed.")

    print(
        f"Total detections: {len(all_detections)}"
    )

    print(
        f"Raw person detections: {raw_person_count}"
    )

    print(
        f"Filtered person detections: "
        f"{len(filtered_person_detections)}"
    )

    print(
        f"Suppressed person detections: "
        f"{len(possible_duplicate_persons)}"
    )

    print(f"Annotated image: {image_output}")

    print(f"Detection report: {json_output}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()





