from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    ROOT / "research"
    / "independent_scenario_evaluation"
    / "candidate_images"
)

DETECTOR_DIR = (
    ROOT / "reports"
    / "independent_scenario_evaluation"
    / "predictions_current"
)

OUTPUT_DIR = ROOT / "reports" / "reasoning_model_comparison"
RAW_DIR = OUTPUT_DIR / "raw_outputs"

ANNOTATION_FILE = (
    ROOT / "research"
    / "independent_scenario_evaluation"
    / "independent_human_annotations_clean.csv"
)

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    "minicpm-v:latest",
    "qwen3-vl:4b-instruct",
]

IMAGE_IDS = [
    "CAND-007.jpg",
    "CDC-001.jpg",
    "CDC-002.jpg",
    "CDC-003.jpg",
    "CDC-004.jpg",
    "MASSDOT-001.jpg",
]

CATEGORIES = [
    "potential_missing_ppe",
    "potential_worker_machinery_interaction",
]

PROMPT_VERSION = "reasoning_comparison_v4_2_explicit_visual_ppe_rules"

MAX_IMAGE_DIMENSION = 1600
IMAGE_JPEG_QUALITY = 88

SYSTEM_PROMPT = """
You are a cautious visual screening assistant evaluating
construction-site images.

You receive an original image and detector-generated evidence.
Detector predictions are fallible and are not ground truth.

Evaluate these categories independently:
1. potential_missing_ppe
2. potential_worker_machinery_interaction

GENERAL EVIDENCE RULES:

- Inspect the original image yourself.
- Separate direct visual observations from detector predictions.
- visual_evidence may contain only details directly visible in the image.
- Never copy detector class names into visual_evidence as if confirmed.
- Do not invent objects, clothing, body parts, or actions.
- An empty visual_evidence array is acceptable.
- Do not infer absence from an area that is obscured, too small,
  out of frame, or otherwise impossible to inspect.
- Do not treat a detector box as proof that its detected object
  or condition is present.
- Do not use evidence from one category to justify another.

CATEGORY 1: potential_missing_ppe

Assess each PPE item independently. For every item below,
return exactly one status:

- present: the item is visibly identifiable.
- potentially_missing: the relevant body area is sufficiently
  visible to assess, and the required item is not visibly present
  or a clearly different item is worn.
- not_assessable: the relevant body area is obscured, out of frame,
  too small, or otherwise cannot be reliably inspected.
- uncertain: the area is visible, but the visual evidence is
  genuinely ambiguous and does not support either presence or absence.

IMPORTANT ITEM-LEVEL VISUAL RULES:
- Assess visibility of each body area before deciding PPE status.
- If the head is clearly visible and the person wears an ordinary
  baseball cap or soft cap rather than a construction hard hat,
  classify helmet as potentially_missing. Do not call this
  not_assessable merely because the cap is not identifiable as PPE.
- If the torso is clearly visible and shows an ordinary shirt or
  jacket with no visible high-visibility garment, classify
  high_visibility_vest as potentially_missing.
- If the hands are hidden, classify gloves as not_assessable.
- If the feet or lower legs are outside the frame or obscured,
  classify boots as not_assessable.
- Do not use not_assessable when the relevant body area is visible
  and the absence of the required item can be visually assessed.
- Do not use potentially_missing when the relevant body area itself
  cannot be inspected.
- Explain the actual visible evidence in visual_basis.


Assess these four items:
- helmet
- high_visibility_vest
- gloves
- boots

For each item:
- Describe the visible body area and what can actually be seen.
- Do not mark gloves missing when hands are obscured.
- Do not mark boots missing when feet or lower legs are obscured.
- A soft cap or baseball-style cap is not a construction hard hat.
- Do not assume a vest is absent if the torso cannot be inspected.
- A no_* detector prediction does not prove PPE is absent.
- Do not infer a regulatory or site-policy violation.
- Do not infer that all PPE is required by a policy unless
  that policy is explicitly supplied.

Set the category decision using these rules:
- supported: at least one item is potentially_missing based on
  direct visual evidence.
- not_supported: all four items can be reliably assessed and none
  is potentially_missing.
- uncertain: no item is established as potentially_missing and
  at least one item is not_assessable or uncertain.

The item-level assessments take priority over broad statements.
Do not describe an obscured item as absent.

CATEGORY 2: potential_worker_machinery_interaction

Assess only visible evidence of a potential interaction.

- Identify the person and machinery, if possible.
- Describe observable positions and actions separately.
- Being an operator or sitting in a machine does not by itself
  establish a hazardous interaction.
- A person near machinery does not establish a hazardous interaction.
- Bounding-box overlap does not prove physical contact.
- Image-space proximity does not establish real-world distance.
- Do not assume machinery is operating.
- Do not infer a person's task, danger, or exclusion-zone boundary.
- Use uncertain when the image does not establish an interaction.

For each category, choose exactly one decision:
- supported
- not_supported
- uncertain

For worker-machinery interaction:
- supported requires direct visual evidence of an observable
  interaction, not proximity alone or a detector relationship.
- not_supported means sufficient visible evidence does not support
  the potential interaction.
- uncertain means the image or context is insufficient.

IMPORTANT LIMITATIONS:

- Do not call a site safe because a condition is not supported.
- Do not claim physical distance from image-space measurements.
- Do not claim physical contact from bounding-box overlap.
- Do not infer machine operation.
- Do not treat a model decision as safety certification.

Return ONLY valid JSON using this exact structure:

{
  "potential_missing_ppe": {
    "decision": "uncertain",
    "ppe_item_assessments": [
      {
        "item": "helmet",
        "status": "not_assessable",
        "visual_basis": ""
      },
      {
        "item": "high_visibility_vest",
        "status": "not_assessable",
        "visual_basis": ""
      },
      {
        "item": "gloves",
        "status": "not_assessable",
        "visual_basis": ""
      },
      {
        "item": "boots",
        "status": "not_assessable",
        "visual_basis": ""
      }
    ],
    "visual_evidence": [],
    "detector_evidence_assessment": "",
    "uncertainty": ""
  },
  "potential_worker_machinery_interaction": {
    "decision": "uncertain",
    "visual_evidence": [],
    "detector_evidence_assessment": "",
    "uncertainty": ""
  },
  "overall_limitations": [],
  "unsupported_claims_to_avoid": []
}

Every PPE item must appear exactly once in ppe_item_assessments.
Use only the four allowed PPE statuses.
Each visual_basis must explain the visible evidence or why the
item cannot be assessed.
visual_evidence must contain category-specific direct observations.
detector_evidence_assessment must discuss detector predictions separately.
uncertainty must describe unresolved image or contextual limitations.
Use only supported, not_supported, or uncertain for decisions.
""".strip()


def load_annotations() -> list[dict[str, str]]:
    """Read labels for validation only. Never send labels to models."""

    with ANNOTATION_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    required = {
        "image_id",
        "finding_category",
        "ground_truth",
    }

    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError("Unexpected annotation CSV structure.")

    return rows


def validate_inputs(image_ids: list[str]) -> None:
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not DETECTOR_DIR.is_dir():
        raise FileNotFoundError(
            f"Detector directory not found: {DETECTOR_DIR}"
        )

    if not ANNOTATION_FILE.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    annotations = load_annotations()

    for image_id in image_ids:
        image_path = IMAGE_DIR / image_id
        detector_path = (
            DETECTOR_DIR / Path(image_id).stem / "detections.json"
        )

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        if not detector_path.is_file():
            raise FileNotFoundError(
                f"Missing detector evidence: {detector_path}"
            )

        for category in CATEGORIES:
            matches = [
                row for row in annotations
                if row["image_id"] == image_id
                and row["finding_category"] == category
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one annotation for "
                    f"{image_id}, {category}; found {len(matches)}."
                )

            if matches[0]["ground_truth"] not in {
                "0", "1", "uncertain"
            }:
                raise ValueError(
                    f"Invalid label for {image_id}, {category}."
                )


def load_detector_evidence(image_id: str) -> dict[str, Any]:
    path = (
        DETECTOR_DIR / Path(image_id).stem / "detections.json"
    )

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    source_detections = report.get(
        "analysis_detections",
        report.get("detections", []),
    )

    detections = []

    for item in source_detections[:40]:
        detections.append({
            "detector": item.get("detector"),
            "class_name": item.get("class_name"),
            "confidence": item.get("confidence"),
            "box_xyxy": item.get("box_xyxy"),
        })

    inventory = report.get("scene_inventory", {})
    spatial = report.get("spatial_analysis", {})

    relationships = []

    for item in spatial.get("relationships", [])[:10]:
        relationships.append({
            "predicate": item.get("predicate"),
            "source_id": item.get("source_id"),
            "target_id": item.get("target_id"),
            "evidence": item.get("evidence"),
        })

    scene_graph = report.get("scene_graph", {})

    return {
        "image_dimensions": scene_graph.get("image_dimensions"),
        "confidence_threshold": report.get("confidence_threshold"),
        "total_detections": report.get("total_detections"),
        "detections": detections,
        "category_counts": inventory.get("category_counts", {}),
        "spatial_summary": {
            "person_count": spatial.get("person_count", 0),
            "machinery_count": spatial.get("machinery_count", 0),
            "relationship_count": spatial.get("relationship_count", 0),
            "relationships": relationships,
        },
        "interpretation_limits": [
            "Detector predictions are fallible.",
            "A no_* PPE prediction does not prove PPE is absent.",
            "Image-space proximity does not establish physical distance.",
            "Bounding-box overlap does not prove physical contact.",
            "The image may not establish machine operation or site requirements.",
        ],
    }


def encode_image(image_path: Path) -> str:
    """Resize the image and encode it as JPEG for Ollama."""

    with Image.open(image_path) as original:
        image = original.convert("RGB")

        image.thumbnail(
            (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            Image.Resampling.LANCZOS,
        )

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=IMAGE_JPEG_QUALITY,
            optimize=True,
        )

    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_user_prompt(
    image_id: str,
    evidence: dict[str, Any],
) -> str:
    compact_json = json.dumps(
        evidence,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return f"""
Image ID: {image_id}

Inspect the original image carefully.

Evaluate the following categories independently:
1. potential_missing_ppe
2. potential_worker_machinery_interaction

DETECTOR-GENERATED EVIDENCE:
{compact_json}

For potential_missing_ppe, assess helmet, high_visibility_vest,
gloves, and boots separately. For each item, identify whether the
relevant body area is visible and choose the correct item status.
Do not mark an item potentially_missing when its body area is obscured.
When a body area is visible, apply the item-level visual rules:
an ordinary cap instead of a hard hat can support potentially_missing;
a visible ordinary shirt with no high-visibility garment can support
potentially_missing. Do not downgrade visible absence to not_assessable.

For potential_worker_machinery_interaction, describe the visible
person, machinery, and observable action separately. Proximity or
operator status alone does not establish a hazardous interaction.

For both categories:
1. Describe only direct visual observations.
2. Assess detector evidence separately.
3. Explain uncertainty.
4. Select supported, not_supported, or uncertain.

Do not use detector predictions as visual observations.
Do not invent evidence when the image is unclear.
Return only the exact JSON structure specified in the system instructions.
""".strip()



def extract_json(response_text: str) -> dict[str, Any]:
    """Parse and validate the v4.1 item-level PPE response schema."""

    text_value = response_text.strip()

    if text_value.startswith("```"):
        text_value = re.sub(
            r"^```(?:json)?\s*",
            "",
            text_value,
            flags=re.IGNORECASE,
        )
        text_value = re.sub(r"\s*```$", "", text_value)

    try:
        result = json.loads(text_value)
    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            text_value,
            flags=re.DOTALL,
        )
        if not match:
            raise ValueError("Model did not return a JSON object.")
        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object.")

    allowed_decisions = {
        "supported",
        "not_supported",
        "uncertain",
    }

    # Normalize evidence objects into strings without inventing evidence.
    for category in CATEGORIES:
        item = result.get(category)

        if not isinstance(item, dict):
            raise ValueError(
                f"Missing response category: {category}"
            )

        if item.get("decision") not in allowed_decisions:
            raise ValueError(
                f"Invalid decision for {category}: "
                f"{item.get('decision')}"
            )

        evidence = item.get("visual_evidence")

        if not isinstance(evidence, list):
            raise ValueError(
                f"visual_evidence must be an array for {category}."
            )

        normalized_evidence = []

        for entry in evidence:
            if isinstance(entry, str):
                normalized_evidence.append(entry)
            elif (
                isinstance(entry, dict)
                and isinstance(entry.get("description"), str)
            ):
                normalized_evidence.append(entry["description"])
            else:
                raise ValueError(
                    f"Invalid visual_evidence entry for {category}."
                )

        item["visual_evidence"] = normalized_evidence

        if not isinstance(
            item.get("detector_evidence_assessment"), str
        ):
            raise ValueError(
                f"detector_evidence_assessment must be a string "
                f"for {category}."
            )

        if "uncertainty" not in item:
            item["uncertainty"] = ""
        elif not isinstance(item["uncertainty"], str):
            raise ValueError(
                f"uncertainty must be a string for {category}."
            )
    # Validate the four independent PPE assessments.
    ppe = result["potential_missing_ppe"]
    assessments = ppe.get("ppe_item_assessments")

    expected_items = {
        "helmet",
        "high_visibility_vest",
        "gloves",
        "boots",
    }

    allowed_statuses = {
        "present",
        "potentially_missing",
        "not_assessable",
        "uncertain",
    }

    if not isinstance(assessments, list):
        raise ValueError(
            "ppe_item_assessments must be an array."
        )

    if len(assessments) != len(expected_items):
        raise ValueError(
            "Exactly four PPE item assessments are required."
        )

    seen_items = set()

    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError(
                "Each PPE assessment must be an object."
            )

        item_name = assessment.get("item")
        status = assessment.get("status")
        visual_basis = assessment.get("visual_basis")

        if item_name not in expected_items:
            raise ValueError(
                f"Unexpected PPE item: {item_name}"
            )

        if item_name in seen_items:
            raise ValueError(
                f"Duplicate PPE item assessment: {item_name}"
            )

        seen_items.add(item_name)

        if status not in allowed_statuses:
            raise ValueError(
                f"Invalid PPE status for {item_name}: {status}"
            )

        if not isinstance(visual_basis, str):
            raise ValueError(
                f"visual_basis must be a string for {item_name}."
            )

    if seen_items != expected_items:
        raise ValueError(
            "One or more PPE items are missing."
        )

    statuses = {
        assessment["status"]
        for assessment in assessments
    }

    if "potentially_missing" in statuses:
        expected_decision = "supported"
    elif statuses == {"present"}:
        expected_decision = "not_supported"
    else:
        expected_decision = "uncertain"

    actual_decision = ppe["decision"]
    ppe["decision_consistency"] = {
        "consistent": actual_decision == expected_decision,
        "expected_decision_from_item_assessments": expected_decision,
        "model_decision": actual_decision,
        "warning": (
            ""
            if actual_decision == expected_decision
            else (
                "The model decision conflicts with its item-level "
                "PPE assessments."
            )
        ),
    }

    for field in (
        "overall_limitations",
        "unsupported_claims_to_avoid",
    ):
        if not isinstance(result.get(field), list):
            raise ValueError(
                f"{field} must be an array."
            )

        if not all(
            isinstance(entry, str)
            for entry in result[field]
        ):
            raise ValueError(
                f"Every {field} entry must be a string."
            )

    return result

def call_model(
    model: str,
    image_base64: str,
    user_prompt: str,
) -> tuple[str, dict[str, Any]]:

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
                "images": [image_base64],
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 1024,
        },
    }

    # Qwen3-VL supports a separate thinking-mode setting.
    # Disable it so the API returns the final answer in message.content.
    if "qwen3-vl" in model.lower():
        payload["think"] = False

    started = time.perf_counter()

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=900,
        )

        response.raise_for_status()

    except requests.HTTPError as exc:
        details = response.text[:2000]

        raise RuntimeError(
            f"Ollama HTTP error: {exc}. "
            f"Server response: {details}"
        ) from exc

    elapsed = time.perf_counter() - started
    body = response.json()

    message = body.get("message", {})
    response_text = message.get("content", "")

    if not isinstance(response_text, str) or not response_text.strip():
        thinking_present = bool(message.get("thinking"))

        raise ValueError(
            "Ollama returned empty final content. "
            f"Thinking mode requested off for Qwen: "
            f"{'think' in payload}. "
            f"Thinking field present: {thinking_present}. "
            f"Done reason: {body.get('done_reason')}. "
            "The thinking field is not being used as the final answer."
        )

    metadata = {
        "elapsed_seconds": round(elapsed, 3),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_count": body.get("eval_count"),
        "total_duration_ns": body.get("total_duration"),
        "load_duration_ns": body.get("load_duration"),
        "prompt_eval_duration_ns": body.get("prompt_eval_duration"),
        "eval_duration_ns": body.get("eval_duration"),
        "thinking_disabled": "think" in payload,
    }

    return response_text, metadata


def run_experiment(
    selected_images: list[str],
    selected_models: list[str],
) -> None:

    validate_inputs(selected_images)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "models": selected_models,
        "images": selected_images,
        "categories": CATEGORIES,
        "temperature": 0,
        "seed": 42,
        "num_ctx": 8192,
        "num_predict": 1024,
        "max_image_dimension": MAX_IMAGE_DIMENSION,
        "image_jpeg_quality": IMAGE_JPEG_QUALITY,
        "qwen_thinking_disabled": True,
        "annotation_file": str(ANNOTATION_FILE.relative_to(ROOT)),
        "detector_source": str(DETECTOR_DIR.relative_to(ROOT)),
        "human_annotations_sent_to_models": False,
        "v4_schema": {
            "visual_evidence": "category-specific direct image observations only",
            "detector_evidence_assessment": "separate detector assessment",
            "empty_visual_evidence_allowed": True,
            "schema_validation": True,
        },
    }

    manifest_path = OUTPUT_DIR / f"manifest_{run_id}.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    summary_rows = []

    total_runs = len(selected_images) * len(selected_models)
    completed_runs = 0

    for image_id in selected_images:
        print(f"\nImage: {image_id}")

        image_path = IMAGE_DIR / image_id
        evidence = load_detector_evidence(image_id)
        user_prompt = build_user_prompt(image_id, evidence)
        image_base64 = encode_image(image_path)

        print(f"  Compact prompt: {len(user_prompt):,} characters")
        print(f"  Resized image: maximum {MAX_IMAGE_DIMENSION} pixels")

        for model in selected_models:
            completed_runs += 1

            print(
                f"\n  [{completed_runs}/{total_runs}] "
                f"Running {model} ..."
            )

            raw_text = ""
            metadata = {}

            try:
                raw_text, metadata = call_model(
                    model,
                    image_base64,
                    user_prompt,
                )

                parsed = extract_json(raw_text)
                status = "SUCCESS"
                error = None

            except Exception as exc:
                parsed = None
                status = "ERROR"
                error = f"{type(exc).__name__}: {exc}"

                print(f"    ERROR: {error}")

            model_key = re.sub(
                r"[^a-zA-Z0-9_-]+",
                "_",
                model,
            )

            output = {
                "run_id": run_id,
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "image_id": image_id,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "status": status,
                "metadata": metadata,
                "raw_response": raw_text,
                "parsed_response": parsed,
                "error": error,
            }

            output_path = (
                RAW_DIR
                / run_id
                / model_key
                / f"{Path(image_id).stem}.json"
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path.write_text(
                json.dumps(
                    output,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary_rows.append({
                "run_id": run_id,
                "image_id": image_id,
                "model": model,
                "status": status,
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "output_file": str(output_path.relative_to(ROOT)),
                "error": error,
            })

            if status == "SUCCESS":
                print(
                    f"    SUCCESS: "
                    f"{metadata['elapsed_seconds']} seconds"
                )

    summary_path = OUTPUT_DIR / f"run_summary_{run_id}.csv"

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_id",
                "image_id",
                "model",
                "status",
                "elapsed_seconds",
                "output_file",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    success_count = sum(
        row["status"] == "SUCCESS"
        for row in summary_rows
    )

    print("\nExperiment finished.")
    print(
        f"Successful responses: "
        f"{success_count}/{len(summary_rows)}"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Raw outputs: {RAW_DIR / run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local vision-language models "
            "on construction-site reasoning."
        )
    )

    parser.add_argument(
        "--image",
        choices=IMAGE_IDS,
        help="Run one selected image.",
    )

    parser.add_argument(
        "--model",
        choices=MODELS,
        help="Run one selected model.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all six images with both models.",
    )

    args = parser.parse_args()

    if args.all:
        if args.image or args.model:
            parser.error(
                "--all cannot be combined with --image or --model."
            )

        selected_images = IMAGE_IDS
        selected_models = MODELS

    else:
        if not args.image or not args.model:
            parser.error(
                "Specify both --image and --model, or use --all."
            )

        selected_images = [args.image]
        selected_models = [args.model]

    run_experiment(
        selected_images,
        selected_models,
    )


if __name__ == "__main__":
    main()
