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

PROMPT_VERSION = "reasoning_comparison_v4_evidence_grounding"

MAX_IMAGE_DIMENSION = 1600
IMAGE_JPEG_QUALITY = 88

SYSTEM_PROMPT = """
You are a cautious visual screening assistant evaluating
construction-site images.

You receive:
1. An original construction-site image.
2. Detector-generated evidence describing machine predictions.

Detector predictions are fallible. They are not ground truth.

Evaluate these two categories independently:

1. potential_missing_ppe
2. potential_worker_machinery_interaction

For EACH category, separate three things:

A. DIRECT VISUAL OBSERVATIONS
Describe only objects, people, clothing, and conditions that
you can personally identify in the original image.

B. DETECTOR EVIDENCE ASSESSMENT
Explain whether the supplied detector predictions are relevant,
consistent with the image, uncertain, or insufficient.

C. CATEGORY DECISION
Decide whether the available visual evidence supports a
potential condition.

VISUAL EVIDENCE RULES:

- visual_evidence must contain only observations from the original image.
- Do not copy detector class names into visual_evidence as if
  they were visually confirmed.
- Each observation must describe a specific visible object,
  clothing item, or condition relevant to that category.
- Do not use observations from one category to justify the other.
- Do not infer that an object is absent simply because it
  cannot be identified.
- If no relevant visual observation can be reliably made,
  return an empty visual_evidence array.
- Never invent observations to fill the array.
- If an observation is ambiguous, explicitly describe the ambiguity.
- Do not treat a detector bounding box as proof that the
  detected object or condition is actually present.

CATEGORY 1: potential_missing_ppe

Assess visible evidence of potentially missing personal
protective equipment.

- Identify the person and the specific PPE item, if visible.
- State whether the relevant body area and PPE status can
  actually be inspected.
- A visible person without a clearly identifiable helmet,
  vest, gloves, boots, or goggles is not automatically proof
  that the equipment is absent.
- A no_* detector prediction does not prove PPE is absent.
- Do not infer site-policy violations unless the applicable
  policy is explicitly supplied.
- If the image cannot establish the PPE status, use uncertain.

CATEGORY 2: potential_worker_machinery_interaction

Assess visible evidence of a potential worker-machinery
interaction.

- Identify the visible person and machinery, if possible.
- Describe their observable positions and actions.
- Distinguish proximity from actual interaction.
- A person near machinery does not establish interaction.
- Bounding-box overlap does not prove physical contact.
- Image-space proximity does not establish real-world distance.
- Do not assume machinery is operating.
- Do not infer a person's task or an exclusion-zone boundary.
- If interaction cannot be established from visible evidence,
  use uncertain.

DECISIONS:

Choose exactly one for each category:

- supported: visible image evidence supports a potential condition.
- not_supported: the image provides sufficient relevant visual
  evidence to conclude that the potential condition is not supported.
- uncertain: image quality, occlusion, ambiguity, or missing
  context prevents a reliable decision.

Do not use not_supported merely because evidence is missing.
Do not use supported based only on detector predictions.

IMPORTANT LIMITATIONS:

- Inspect the original image yourself.
- Do not automatically accept detector predictions.
- Do not invent objects, clothing, actions, or details.
- Do not claim physical distance from image-space measurements.
- Do not claim physical contact from bounding-box overlap.
- Do not infer machine operation.
- Do not call a site safe because a condition is not supported.
- Do not allow one category's evidence to determine the other.
- Do not treat a model decision as a safety certification.

Return ONLY valid JSON using this exact structure:

{
  "potential_missing_ppe": {
    "decision": "uncertain",
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

For each category:

- visual_evidence must be a JSON array of strings.
- Each string must describe a specific direct visual observation.
- detector_evidence_assessment must discuss detector evidence separately.
- uncertainty must explain unresolved visual or contextual limitations.
- Use an empty array when no relevant direct visual evidence
  can be reliably identified.

Use only supported, not_supported, or uncertain as decisions.
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

For each category, follow this order:

1. Describe only what you can directly observe in the image.
2. Assess the detector evidence separately.
3. Explain whether the visual observations support the category.
4. Select supported, not_supported, or uncertain.

Do not use detector predictions as visual observations.

Do not invent evidence when the image is unclear.
An empty visual_evidence array is acceptable when no relevant
visual observation can be reliably made.

Do not interpret proximity as proven interaction.
Do not interpret a no_* prediction as proof of missing PPE.

Return only the exact JSON structure specified in the
system instructions.
""".strip()


def extract_json(response_text: str) -> dict[str, Any]:
    """Parse and validate the v4 response schema."""

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

    for category in CATEGORIES:
        item = result.get(category)

        if not isinstance(item, dict):
            raise ValueError(
                f"Missing response category: {category}"
            )

        decision = item.get("decision")

        if decision not in {
            "supported",
            "not_supported",
            "uncertain",
        }:
            raise ValueError(
                f"Invalid decision for {category}: {decision}"
            )

        visual_evidence = item.get("visual_evidence")

        if not isinstance(visual_evidence, list):
            raise ValueError(
                f"visual_evidence must be an array for {category}."
            )

        if not all(
            isinstance(entry, str)
            for entry in visual_evidence
        ):
            raise ValueError(
                f"Every visual_evidence entry must be a string "
                f"for {category}."
            )

        for field in (
            "detector_evidence_assessment",
            "uncertainty",
        ):
            if not isinstance(item.get(field), str):
                raise ValueError(
                    f"{field} must be a string for {category}."
                )

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
