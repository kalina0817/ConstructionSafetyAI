
import base64
import json
import os
import shutil
import subprocess
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageOps

from screening_assessment import (
    PPE_ITEMS,
    SCREENING_SYSTEM_PROMPT,
    parse_visual_screening_output,
)


OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = {
    "MiniCPM-V": "minicpm-v:latest",
    "Qwen3-VL": "qwen3-vl:4b-instruct",
}

PROMPT_VERSION = "reasoning_comparison_v4_3_five_ppe_legacy_compatible"
MAX_IMAGE_DIMENSION = 1600
IMAGE_JPEG_QUALITY = 88

CATEGORIES = (
    "potential_missing_ppe",
    "potential_worker_machinery_interaction",
)

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
- Assess the eyes and face area for construction safety glasses
  or goggles. Ordinary eyeglasses are not proof of safety-rated
  eye protection.
- If the eyes are clearly visible and no protective eyewear is
  visible, classify eye_protection as potentially_missing.
- If safety glasses or goggles are visibly identifiable,
  classify eye_protection as present.
- If the eyes are obscured, too small, out of frame, or the
  eyewear cannot be distinguished, use not_assessable or
  uncertain as appropriate.
- Do not use not_assessable when the relevant body area is visible
  and the absence of the required item can be visually assessed.
- Do not use potentially_missing when the relevant body area itself
  cannot be inspected.
- Explain the actual visible evidence in visual_basis.

Assess these five items:
- helmet
- high_visibility_vest
- gloves
- boots
- eye_protection

For each item:
- Describe the visible body area and what can actually be seen.
- Do not mark gloves missing when hands are obscured.
- Do not mark boots missing when feet or lower legs are obscured.
- Do not treat ordinary eyeglasses as confirmed construction
  safety eyewear.
- Do not mark eye protection missing when the eyes cannot be
  reliably inspected.
- A soft cap or baseball-style cap is not a construction hard hat.
- Do not assume a vest is absent if the torso cannot be inspected.
- A no_* detector prediction does not prove PPE is absent.
- Do not infer a regulatory or site-policy violation.
- Do not infer that all PPE is required by a policy unless
  that policy is explicitly supplied.

Set the category decision using these rules:
- supported: at least one item is potentially_missing based on
  direct visual evidence.
- not_supported: all five items can be reliably assessed and none
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

OUTPUT LENGTH RULES:

- Keep every visual_basis to at most two concise sentences.
- Keep each detector_evidence_assessment to at most two sentences.
- Do not repeat detector predictions or explanations.
- Mention each relevant detector class at most once per category.
- Keep each uncertainty explanation to one concise sentence.
- Use empty arrays when there are no direct visual observations.
- Do not add commentary outside the required JSON.

Return ONLY valid JSON using this exact structure:

{
  "potential_missing_ppe": {
    "decision": "uncertain",
    "ppe_item_assessments": [
      {"item": "helmet", "status": "not_assessable", "visual_basis": ""},
      {"item": "high_visibility_vest", "status": "not_assessable", "visual_basis": ""},
      {"item": "gloves", "status": "not_assessable", "visual_basis": ""},
      {"item": "boots", "status": "not_assessable", "visual_basis": ""},
      {"item": "eye_protection", "status": "not_assessable", "visual_basis": ""}
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


def encode_image(image_path: Path) -> str:
    """Resize and encode the original image for Ollama."""

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


def build_detector_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Build compact detector, spatial, and preliminary screening evidence."""

    source_detections = report.get(
        "analysis_detections",
        report.get("detections", []),
    )

    if not isinstance(source_detections, list):
        source_detections = []

    detections = []

    for item in source_detections[:40]:
        if not isinstance(item, dict):
            continue

        detections.append({
            "detector": item.get("detector"),
            "model_variant": item.get("model_variant"),
            "class_name": item.get("class_name"),
            "confidence": item.get("confidence"),
            "box_xyxy": item.get("box_xyxy"),
        })

    inventory = report.get("scene_inventory", {})
    if not isinstance(inventory, dict):
        inventory = {}

    spatial = report.get("spatial_analysis", {})
    if not isinstance(spatial, dict):
        spatial = {}

    relationships = spatial.get("relationships", [])
    if not isinstance(relationships, list):
        relationships = []

    compact_relationships = []

    for item in relationships[:10]:
        if not isinstance(item, dict):
            continue

        compact_relationships.append({
            "predicate": item.get("predicate"),
            "person_class": item.get("person_class"),
            "machinery_class": item.get("machinery_class"),
            "status": item.get("status"),
            "center_distance_pixels": item.get("center_distance_pixels"),
            "normalized_image_distance": item.get(
                "normalized_image_distance"
            ),
            "evidence": item.get("evidence"),
        })

    safety = report.get("safety_assessment", {})
    if not isinstance(safety, dict):
        safety = {}

    findings = safety.get("findings", [])
    if not isinstance(findings, list):
        findings = []

    compact_findings = []

    finding_fields = (
        "finding_id",
        "category",
        "rule_id",
        "status",
        "alert_generated_automatically",
        "source_object_ids",
        "relationship_id",
        "evidence",
        "observation",
        "limitation",
        "uncertainty_reasons",
        "recommendation",
    )

    for finding in findings[:20]:
        if not isinstance(finding, dict):
            continue

        compact_findings.append({
            key: finding.get(key)
            for key in finding_fields
            if key in finding
        })

    screening = report.get("screening_outcome", {})
    if not isinstance(screening, dict):
        screening = {}

    compact_screening = {
        key: screening.get(key)
        for key in (
            "outcome",
            "explanation",
            "finding_count",
            "detector_coverage",
            "high_confidence_alert_available",
            "high_confidence_alert_limitation",
            "interpretation_limit",
        )
        if key in screening
    }

    scene_graph = report.get("scene_graph", {})
    if not isinstance(scene_graph, dict):
        scene_graph = {}

    return {
        "evidence_source": (
            "Detector predictions and rule-generated screening findings; "
            "these are not ground truth."
        ),
        "image_dimensions": scene_graph.get("image_dimensions"),
        "confidence_threshold": report.get("confidence_threshold"),
        "total_detections": report.get("total_detections"),
        "detections": detections,
        "category_counts": inventory.get("category_counts", {}),
        "spatial_summary": {
            "person_count": spatial.get("person_count", 0),
            "machinery_count": spatial.get("machinery_count", 0),
            "relationship_count": spatial.get("relationship_count", 0),
            "relationships": compact_relationships,
        },
        "preliminary_safety_findings": compact_findings,
        "preliminary_screening_outcome": compact_screening,
        "safety_assessment_interpretation_limit": safety.get(
            "interpretation_limit"
        ),
        "unassessed_conditions": safety.get(
            "unassessed_conditions", []
        ),
        "interpretation_limits": [
            "Detector predictions are fallible.",
            "A no_* PPE prediction does not prove PPE is absent.",
            "Image-space proximity does not establish physical distance.",
            "Bounding-box overlap does not prove physical contact.",
            "The image may not establish machine operation or site requirements.",
            "Detector relationships are not proof of a hazardous interaction.",
            "Preliminary alerts are screening findings, not confirmed violations.",
        ],
    }

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

DETECTOR, SPATIAL, AND PRELIMINARY SCREENING EVIDENCE:
{compact_json}

The evidence includes preliminary safety findings generated by the
application's earlier detection and screening stages. Evaluate those
findings against the original image. Do not treat a preliminary finding
as ground truth.

For potential_missing_ppe, assess all five items separately:
helmet, high_visibility_vest, gloves, boots, and eye_protection.
Return exactly one item-level assessment for each item. For each item,
identify whether its relevant body area is visible and choose the
correct status. Do not mark an item potentially_missing when its body
area is obscured. When a body area is visible, apply the item-level
visual rules: an ordinary cap instead of a hard hat can support
potentially_missing; a visible ordinary shirt with no high-visibility
garment can support potentially_missing. Assess eye_protection
independently using the eyes and face area. Do not downgrade visible absence to not_assessable.

Assess eye protection as its own PPE item, including visible
construction safety glasses or goggles. Ordinary eyeglasses alone
do not establish that construction eye protection is present. If
the eyes cannot be inspected clearly, mark eye_protection as
not_assessable or uncertain. Compare eye-protection findings only
with the eye_protection item assessment, not with other PPE items.

For potential_worker_machinery_interaction, describe the visible
person, machinery, and observable action separately. Evaluate the
specific preliminary finding and its spatial evidence. Proximity,
bounding-box overlap, or operator status alone does not establish a
hazardous interaction or real-world distance.

For both categories:
1. Describe only direct visual observations.
2. Assess detector and preliminary screening evidence separately.
3. State whether the image supports, conflicts with, or cannot assess
   the relevant preliminary finding.
4. Explain uncertainty and limitations.
5. Select supported, not_supported, or uncertain.

A not_supported decision is a disagreement with a preliminary model
finding, not proof that the scene is safe. Do not erase, rewrite, or
silently override any preliminary alert.

Do not use detector predictions as visual observations.
Do not invent evidence when the image is unclear.
Return only the exact JSON structure specified in the system instructions.
""".strip()


def extract_json(response_text: str) -> dict[str, Any]:
    """Parse and validate the v4.2 response schema."""

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
        match = re.search(r"\{.*\}", text_value, flags=re.DOTALL)
        if not match:
            raise ValueError("Model did not return a JSON object.")
        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object.")

    allowed_decisions = {"supported", "not_supported", "uncertain"}

    for category in CATEGORIES:
        item = result.get(category)

        if not isinstance(item, dict):
            raise ValueError(f"Missing response category: {category}")

        if item.get("decision") not in allowed_decisions:
            raise ValueError(
                f"Invalid decision for {category}: {item.get('decision')}"
            )

        evidence = item.get("visual_evidence")
        if not isinstance(evidence, list):
            raise ValueError(
                f"visual_evidence must be an array for {category}."
            )

        normalized = []

        for entry in evidence:
            if isinstance(entry, str):
                normalized.append(entry)
            elif (
                isinstance(entry, dict)
                and isinstance(entry.get("description"), str)
            ):
                normalized.append(entry["description"])
            else:
                raise ValueError(
                    f"Invalid visual_evidence entry for {category}."
                )

        item["visual_evidence"] = normalized

        if not isinstance(item.get("detector_evidence_assessment"), str):
            raise ValueError(
                f"detector_evidence_assessment must be a string for {category}."
            )

        if "uncertainty" not in item:
            item["uncertainty"] = ""
        elif not isinstance(item["uncertainty"], str):
            raise ValueError(
                f"uncertainty must be a string for {category}."
            )

    ppe = result["potential_missing_ppe"]
    assessments = ppe.get("ppe_item_assessments")

    expected_items = {
        "helmet",
        "high_visibility_vest",
        "gloves",
        "boots",
        "eye_protection",
    }

    allowed_statuses = {
        "present",
        "potentially_missing",
        "not_assessable",
        "uncertain",
    }

    legacy_eye_protection_omission = False

    if not isinstance(assessments, list):
        raise ValueError("PPE item assessments must be a list.")

    if len(assessments) == 4:
        if not all(isinstance(item, dict) for item in assessments):
            raise ValueError("Each PPE assessment must be an object.")

        supplied_items = [item.get("item") for item in assessments]

        if (
            len(set(supplied_items)) != 4
            or set(supplied_items) != expected_items - {"eye_protection"}
        ):
            raise ValueError(
                "Legacy PPE responses must contain exactly the four "
                "expected items, with only eye_protection omitted."
            )

        assessments.append({
            "item": "eye_protection",
            "status": "not_assessable",
            "visual_basis": (
                "No eye_protection assessment was returned in the saved "
                "model response. This placeholder was added by the "
                "validator and is not a visual assessment."
            ),
            "assessment_source": "validator_legacy_schema_fallback",
        })

        legacy_eye_protection_omission = True

    elif len(assessments) != 5:
        raise ValueError("Exactly five PPE item assessments are required.")

    seen_items = set()

    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("Each PPE assessment must be an object.")

        item_name = assessment.get("item")
        status = assessment.get("status")
        visual_basis = assessment.get("visual_basis")

        if item_name not in expected_items:
            raise ValueError(f"Unexpected PPE item: {item_name}")

        if item_name in seen_items:
            raise ValueError(f"Duplicate PPE item assessment: {item_name}")

        seen_items.add(item_name)

        if status not in allowed_statuses:
            raise ValueError(
                f"Invalid PPE status for {item_name}: {status}"
            )

        if not isinstance(visual_basis, str):
            raise ValueError(
                f"visual_basis must be a string for {item_name}."
            )

    if legacy_eye_protection_omission:
        ppe["legacy_schema_incomplete"] = True
        ppe["legacy_schema_warning"] = (
            "The saved model response omitted eye_protection. "
            "A not_assessable placeholder was added; it is not a "
            "model-generated visual assessment."
        )

    if seen_items != expected_items:
        raise ValueError("One or more PPE items are missing.")

    statuses = {assessment["status"] for assessment in assessments}

    if "potentially_missing" in statuses:
        expected_decision = "supported"
    elif statuses == {"present"}:
        expected_decision = "not_supported"
    else:
        expected_decision = "uncertain"

    actual_decision = ppe["decision"]
    ppe["decision_consistency"] = {
        "consistent": actual_decision == expected_decision,
        "assessment_set_complete_from_model": not legacy_eye_protection_omission,
        "expected_decision_from_item_assessments": expected_decision,
        "model_decision": actual_decision,
        "decision_corrected": actual_decision != expected_decision,
        "warning": (
            ""
            if actual_decision == expected_decision
            else (
                "The model decision conflicts with its item-level "
                "PPE assessments. The validated decision is derived "
                "from the five item-level statuses."
            )
        ),
    }

    # Keep the VLM's original category decision for model evaluation,
    # while making the schema-consistent decision the canonical value
    # consumed by downstream application logic.
    ppe["model_decision"] = actual_decision
    ppe["decision"] = expected_decision

    for field in ("overall_limitations", "unsupported_claims_to_avoid"):
        if not isinstance(result.get(field), list):
            raise ValueError(f"{field} must be an array.")

        if not all(isinstance(entry, str) for entry in result[field]):
            raise ValueError(f"Every {field} entry must be a string.")

    return result


HEADWEAR_STATUS = {"rigid_hard_hat": "p", "soft_cap": "m", "bare_head": "m",
                   "other_or_unclear": "u"}
POSITION_RELATION = {"seated_on_equipment": "on_machine", "standing_on_equipment": "on_machine",
                     "handling_machine_or_load": "handling_machine_or_load", "beside_equipment": "beside_machine",
                     "in_same_scene_as": "in_same_scene_as", "other_visible_relation": "other_visible_relation", "no_identifiable_equipment": "no_visible_relation",
                     "unclear": "unclear"}


MACHINE_IDENTITIES = ("construction_vehicle", "powered_work_machine", "ordinary_surface_or_hand_tool",
                      "no_machine_visible", "identity_unclear")


BODY_SUPPORT = ("seated_in_operator_station", "feet_on_machine_step_or_deck",
                "independent_surface", "not_visible")
SUPPORT_PROMPT = (
    "Identify what physically supports this worker relative to the machine. Select seated_in_operator_station "
    "only if the worker visibly occupies its seat or operator station; select feet_on_machine_step_or_deck "
    "only when the feet visibly rest on that machine. Seeing a machine part, tracks, controls, or overlapping "
    "boxes does not establish body support. If both subjects appear but the image establishes only co-presence, "
    "select in_same_scene_as. Describe the body-to-machine evidence or the gap in the view. "
)


def grounded_position(observation):
    identity, position = observation["machine_identity"], observation["position"]
    if identity not in MACHINE_IDENTITIES or position not in POSITION_RELATION:
        raise ValueError("Invalid machine identity or worker position.")
    if identity in ("ordinary_surface_or_hand_tool", "no_machine_visible"):
        return "no_identifiable_equipment"
    if identity == "identity_unclear":
        return "unclear"
    support = observation.get("body_support")
    if support == "seated_in_operator_station":
        return "seated_on_equipment"
    if support == "feet_on_machine_step_or_deck":
        return "standing_on_equipment"
    if position in ("seated_on_equipment", "standing_on_equipment"):
        return "in_same_scene_as"
    return position


def focused_worker_view(image, box, machine_boxes):
    """Keep one unmodified worker crop beside a marked full-scene view."""
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid worker box: {box}")
    scene = image.copy()
    pen = ImageDraw.Draw(scene)
    for number, bounds in enumerate(machine_boxes, 1):
        pen.rectangle(tuple(bounds), outline="orange", width=2)
        pen.text((bounds[0]+2, bounds[1]+2), str(number), fill="white", stroke_width=1, stroke_fill="black")
    pen.rectangle((x1, y1, x2-1, y2-1), outline="cyan", width=2)
    crop = image.crop((x1, y1, x2, y2))
    context = ImageOps.contain(scene, (640, 560))
    detail = ImageOps.contain(crop, (360, 560))
    board = Image.new("RGB", (context.width + detail.width + 6, max(context.height, detail.height)), "black")
    board.paste(context, (0, 0))
    board.paste(detail, (context.width + 6, 0))
    buffer = BytesIO()
    board.save(buffer, "JPEG", quality=IMAGE_JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def worker_response_format(pair_ids):
    def fields(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    def choice(values):
        return {"type": "string", "enum": list(values)}

    reason = {"type": "string", "minLength": 1, "maxLength": 160}
    item = fields({"status": choice(["p", "m", "u"]), "reason": reason})
    spatial = {"machine_identity": choice(MACHINE_IDENTITIES), "body_support": choice(BODY_SUPPORT),
               "position": choice(POSITION_RELATION)}
    pair = fields({**spatial, "decision": choice(
        ["supported", "not_supported", "unassessable"]), "visual_evidence": reason, "limitation": reason})
    return fields({"subject": choice(["person", "not_person", "unclear"]),
                   "headwear": fields({"appearance": choice(HEADWEAR_STATUS), "reason": reason}),
                   "ppe": fields({name: item for name in PPE_ITEMS if name != "helmet"}),
                   "equipment_context": fields({**spatial, "reason": reason}),
                   "machinery": fields({pair_id: pair for pair_id in pair_ids})})


def read_worker_observations(raw, worker, pair_ids):
    data = json.loads(raw)
    subject = data["subject"]
    if subject not in ("person", "not_person", "unclear") or set(data["machinery"]) != set(pair_ids):
        raise ValueError("Unexpected worker identity or machinery pairs in model response.")
    head = data["headwear"]
    entries = [[HEADWEAR_STATUS[head["appearance"]], head["reason"]]]
    entries.extend([data["ppe"][name]["status"], data["ppe"][name]["reason"]]
                   for name in PPE_ITEMS if name != "helmet")
    context = data["equipment_context"]
    context["position"] = grounded_position(context)
    if not context["reason"].strip():
        raise ValueError("Invalid scene-context observation.")
    pairs = {}
    for pair_id, judgment in data["machinery"].items():
        position = grounded_position(judgment)
        judgment.pop("position")
        if position == "no_identifiable_equipment":
            judgment["decision"] = "not_supported"
        elif position == "unclear" and judgment["decision"] == "supported":
            judgment["decision"] = "unassessable"
        pairs[pair_id] = {**judgment, "visible_relation": POSITION_RELATION[position]}
    if subject != "person":
        basis = "The close-up does not establish an identifiable worker."
        entries = [["u", basis] for _ in PPE_ITEMS]
        context = {"machine_identity": "identity_unclear", "position": "unclear", "reason": basis}
        for judgment in pairs.values():
            judgment.update(body_support="not_visible", visible_relation="unclear", decision="unassessable", visual_evidence=basis,
                            limitation="A worker-specific relation requires an identifiable worker.")
    analysis = parse_visual_screening_output(json.dumps({"ppe": {worker: entries}, "machinery": pairs}))
    if len(analysis["worker_ppe"]) != len(PPE_ITEMS):
        raise ValueError("Incomplete worker PPE observations.")
    return analysis, {"worker": worker, "subject": subject, **context}


def screen_worker_views(image_path, graph, model, record):
    nodes = {node["object_id"]: node for node in graph["nodes"]}
    workers = [node for node in nodes.values() if node["object_category"] == "person"]
    analysis = {"worker_ppe": [], "machinery_assessments": {}, "worker_context": []}
    raw_outputs = {}
    timings = {}
    if not workers:
        # Keep model selection and availability verifiable on empty detector graphs.
        raw, meta = call_model(model, encode_image(image_path),
            "There are no worker candidates. Return an empty JSON object.", screening=True,
            response_format={"type": "object", "properties": {}, "additionalProperties": False})
        if json.loads(raw) != {}:
            raise ValueError("Unexpected response for an empty worker graph.")
        raw_outputs["scene"], timings["scene"] = raw, meta
    owners = {}
    for edge in graph["relationships"]:
        if edge["predicate"] == "inside_person_bbox":
            owners.setdefault(edge["source_id"], set()).add(edge["target_id"])
    with Image.open(image_path) as original:
        image = original.convert("RGB")
        for node in workers:
            worker = node["object_id"]
            edges = [e for e in graph["relationships"] if e["predicate"] == "near_machinery_in_image"
                     and e["source_id"] == worker]
            ids = [e["relationship_id"] for e in edges]
            targets = [nodes[e["target_id"]] for e in edges]
            view = focused_worker_view(image, node["box_xyxy"], [n["box_xyxy"] for n in targets])
            prompt = (
                "Two views of ONE photo. Left: scene with cyan target box. Right: enlarged pixels inside that box. "
                "Assess ONLY the target. If the crop is machine parts or background, select not_person. "
                "Identify headwear appearance: a rigid protective shell, a soft cloth cap, a bare head, or unclear. "
                "For vest, gloves, boots and eye protection use p visibly present, m inspectable area with visible "
                "evidence of absence, u obscured, cropped or ambiguous. For equipment_context describe the person's "
                "physical position relative to visible equipment anywhere in the scene, even without an orange box. "
                "First identify a construction vehicle or powered work machine by visible physical features. A table, bench, work platform, wood or handheld tool alone is not machinery. If identity is unclear, abstain. "
                "For each numbered orange candidate identify the physical position relative to that specific object, "
                "then independently decide whether a visible configuration supports potential hazardous exposure. "
                "Sitting or proximity alone does not establish that exposure. Never infer motion or real distance. "
                "Give short phrases with directly visible evidence. Orange box numbers and corresponding pair IDs: "
                + str(dict(enumerate(ids, 1))) + ". " + SUPPORT_PROMPT
            )
            raw, meta = call_model(model, view, prompt, screening=True, response_format=worker_response_format(ids))
            raw_outputs[worker], timings[worker] = raw, meta
            record["raw_response"], record["metadata"] = raw_outputs, timings
            observations, context = read_worker_observations(raw, worker, ids)
            # Recheck a rejected subject when distinct, exclusively linked PPE classes
            # independently suggest that a person is present. Keep both model responses.
            if context["subject"] != "person":
                linked_classes = {nodes[obj]["class_name"] for obj, people in owners.items()
                                  if people == {worker} and nodes[obj]["object_category"] == "ppe_prediction"}
                if len(linked_classes) >= 2:
                    retry_prompt = (
                        f"Inspect ONE original photo. Assess only the subject in xyxy {node['box_xyxy']} "
                        f"of original size {image.size}. Determine whether a person is visible there. "
                        "Classify actual headwear and PPE from visible features. Use u for obscured items. "
                        "Identify powered machinery before describing the subject's physical position. "
                        "For a candidate pair judge hazardous exposure separately from position. "
                        f"Pair IDs and machine boxes: {dict(zip(ids, [n['box_xyxy'] for n in targets]))}. "
                        "Return the requested schema with short evidence phrases. " + SUPPORT_PROMPT
                    )
                    retry_raw, retry_meta = call_model(model, encode_image(image_path), retry_prompt,
                        screening=True, response_format=worker_response_format(ids))
                    key = worker + "_original_view"
                    raw_outputs[key], timings[key] = retry_raw, retry_meta
                    observations, context = read_worker_observations(retry_raw, worker, ids)
                    context["identity_check"] = "Original photo rechecked after the crop and linked PPE disagreed."
            if context["subject"] == "person":
                head_nodes = [nodes[obj] for obj, people in owners.items() if people == {worker}
                              and nodes[obj]["class_name"].lower() in ("helmet", "safety helmet", "no_helmet")
                              and nodes[obj].get("box_xyxy")]
                helmet = next(row for row in observations["worker_ppe"] if row["item"] == "helmet")
                if len(head_nodes) == 1:
                    head_node = head_nodes[0]
                    expected = "potentially_missing" if head_node["class_name"].lower() == "no_helmet" else "present"
                    if helmet["status"] != expected:
                        x1, y1, x2, y2 = head_node["box_xyxy"]
                        margin = max(2, round(max(x2-x1, y2-y1) * 0.35))
                        head_view = focused_worker_view(image, (x1-margin, y1-margin, x2+margin, y2+margin), [])
                        prompt = ("Look at the close-up on the right, and use the left scene only for context. "
                                  "What is actually on this head? A rigid construction hard hat has a hard shell. "
                                  "A soft cloth cap has fabric. Choose other_or_unclear if you cannot tell. "
                                  "Describe its visible shape or material. Ignore whether PPE is required.")
                        head_schema = worker_response_format([])["properties"]["headwear"]
                        raw_head, meta_head = call_model(model, head_view, prompt, screening=True,
                                                        response_format=head_schema)
                        key = worker + "_head_detail"
                        raw_outputs[key], timings[key] = raw_head, meta_head
                        detail = json.loads(raw_head)
                        detail_status = {"p": "present", "m": "potentially_missing", "u": "unassessable"}[
                            HEADWEAR_STATUS[detail["appearance"]]]
                        if not detail["reason"].strip():
                            raise ValueError("The headwear detail check returned no visible basis.")
                        helmet["status"] = detail_status if detail_status == expected else "unassessable"
                        helmet["visual_basis"] = "Head detail: " + detail["reason"].strip()
            analysis["worker_ppe"].extend(observations["worker_ppe"])
            analysis["machinery_assessments"].update(observations["machinery_assessments"])
            analysis["worker_context"].append(context)
    record.update(analysis=analysis, raw_response=raw_outputs, metadata={
        "views": timings, "elapsed_seconds": round(sum(view["elapsed_seconds"] for view in timings.values()), 3),
    })


def ensure_local_ollama(model: str) -> None:
    """Start the installed local service if a screening run needs it."""
    version_url = OLLAMA_URL.rsplit("/", 1)[0] + "/version"
    try:
        requests.get(version_url, timeout=3).raise_for_status()
        return
    except requests.ReadTimeout:
        pass  # An existing listener may still be initializing.
    except requests.ConnectionError:
        executable = shutil.which("ollama")
        if not executable and os.name == "nt":
            candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"
            if candidate.is_file():
                executable = str(candidate)
        if not executable:
            raise RuntimeError(f"Ollama is offline and its executable could not be found. Start Ollama to run {model}.")

        subprocess.Popen(
            [executable, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            requests.get(version_url, timeout=2).raise_for_status()
            return
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.5)
    raise RuntimeError(f"Ollama did not become ready at {version_url}. Start Ollama and retry {model}.")


def call_model(
    model: str,
    image_base64: str,
    user_prompt: str,
    *,
    screening: bool = False,
    response_format=None,
) -> tuple[str, dict[str, Any]]:
    """Call Ollama using the v4.2 experiment settings."""

    system_prompt = SCREENING_SYSTEM_PROMPT if screening else SYSTEM_PROMPT
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                "images": [image_base64],
            },
        ],
        "stream": screening,
        "format": response_format or "json",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 4096 if response_format else 8192,
            "num_predict": 2048,
        },
    }

    if "qwen3-vl" in model.lower():
        payload["think"] = False

    if screening:
        ensure_local_ollama(model)

    started = time.perf_counter()

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=screening, timeout=(5, 900)) as response:
            if not response.ok:
                raise RuntimeError(f"Ollama returned HTTP {response.status_code} for {model}: {response.text[:2000]}")
            if screening:
                # Ollama sends progress as NDJSON. The read timeout now measures inactivity.
                parts = []
                body = {}
                for line in response.iter_lines():
                    if not line:
                        continue
                    body = json.loads(line)
                    if body.get("error"):
                        raise RuntimeError(f"Ollama failed while running {model}: {body['error']}")
                    parts.append(body.get("message", {}).get("content", ""))
                    if body.get("done"):
                        break
                if not body.get("done"):
                    raise RuntimeError(f"Ollama ended the response before {model} finished.")
                response_text = "".join(parts)
            else:
                body = response.json()
                response_text = body.get("message", {}).get("content", "")
    except requests.ConnectTimeout as exc:
        raise RuntimeError(f"Timed out connecting to Ollama to run {model}.") from exc
    except requests.ConnectionError as exc:
        raise RuntimeError(f"Ollama connection failed while running {model}: {exc}") from exc

    elapsed = time.perf_counter() - started
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError(
            "Ollama returned empty final content. "
            f"Done reason: {body.get('done_reason')}."
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


def run_vision_language_analysis(
    image_path: str | Path,
    report: dict[str, Any],
    model_label: str,
    output_file: str | Path,
    *,
    screening: bool = False,
) -> dict[str, Any]:
    """
    Run one selected VLM and persist its output independently
    of detector results and automatic alerts.
    """

    if model_label not in MODELS:
        raise ValueError(f"Unsupported vision-language model: {model_label}")

    image_path = Path(image_path)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model_name = MODELS[model_label]

    record: dict[str, Any] = {
        "status": "RUNNING",
        "model_label": model_label,
        "model_name": model_name,
        "prompt_version": "worker_scene_screening_v6" if screening else PROMPT_VERSION,
        "image_path": str(image_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {},
        "analysis": None,
        "raw_response": None,
        "error": None,
    }

    def save_record() -> None:
        output_file.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    save_record()

    try:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if screening:
            screen_worker_views(image_path, report["scene_graph"], model_name, record)
        else:
            user_prompt = build_user_prompt(image_path.name, build_detector_evidence(report))
            raw, metadata = call_model(model_name, encode_image(image_path), user_prompt)
            record["raw_response"], record["metadata"] = raw, metadata
            record["analysis"] = extract_json(raw)

        record["status"] = "COMPLETED"
        save_record()
        return record

    except (ValueError, KeyError, TypeError) as error:
        record["status"] = "VALIDATION_ERROR"
        record["error"] = str(error)
        save_record()
        return record
    except Exception as error:
        record["status"] = "ERROR"
        record["error"] = f"{type(error).__name__}: {error}"
        save_record()
        return record

def build_integrated_assessment(
    report: dict[str, Any],
    vlm_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare each preliminary detector finding with the relevant VLM
    assessment. This function never removes or changes automatic alerts.
    """

    if vlm_result.get("status") != "COMPLETED":
        raise ValueError("A completed VLM result is required.")

    analysis = vlm_result.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("The VLM result does not contain a valid analysis.")

    ppe_result = analysis.get("potential_missing_ppe", {})
    machinery_result = analysis.get(
        "potential_worker_machinery_interaction", {}
    )

    if not isinstance(ppe_result, dict):
        ppe_result = {}
    if not isinstance(machinery_result, dict):
        machinery_result = {}

    ppe_assessments = ppe_result.get("ppe_item_assessments", [])
    if not isinstance(ppe_assessments, list):
        ppe_assessments = []

    ppe_by_item = {
        item.get("item"): item
        for item in ppe_assessments
        if isinstance(item, dict) and isinstance(item.get("item"), str)
    }

    safety = report.get("safety_assessment", {})
    if not isinstance(safety, dict):
        safety = {}

    findings = safety.get("findings", [])
    if not isinstance(findings, list):
        findings = []

    comparisons = []

    ppe_item_terms = {
        "eye_protection": (
            "eye_protection", "eye protection", "goggle",
            "safety glass", "eyewear", "no_goggle", "no_goggles",
        ),
        "high_visibility_vest": (
            "high_visibility_vest", "high visibility vest",
            "safety_vest", "vest",
        ),
        "helmet": (
            "helmet", "hard hat", "hard_hat",
        ),
        "gloves": (
            "gloves", "glove",
        ),
        "boots": (
            "boots", "boot", "safety footwear",
        ),
    }

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        category = str(
            finding.get("finding_category")
            or finding.get("category")
            or ""
        ).lower()
        finding_text = json.dumps(
            {
                "rule_id": finding.get("rule_id"),
                "evidence": finding.get("evidence"),
                "observation": finding.get("observation"),
                "recommendation": finding.get("recommendation"),
            },
            ensure_ascii=False,
        ).lower()

        comparison = {
            "finding_id": finding.get("finding_id"),
            "category": (
                finding.get("finding_category")
                or finding.get("category")
            ),
            "automatic_alert_preserved": bool(
                finding.get("alert_generated_automatically", False)
            ),
            "reasoning_comparison": "not_assessed",
            "matched_ppe_item": None,
            "vlm_status_or_decision": None,
            "explanation": "",
        }

        if "potential_missing_ppe" in category or "ppe" in category:
            matched_item = None

            for item_name, terms in ppe_item_terms.items():
                if any(term in finding_text for term in terms):
                    matched_item = item_name
                    break

            if matched_item is None:
                comparison["explanation"] = (
                    "The preliminary PPE finding could not be matched "
                    "to one specific PPE item. No broad-category "
                    "corroboration was assigned."
                )
            else:
                assessment = ppe_by_item.get(matched_item)

                comparison["matched_ppe_item"] = matched_item

                if assessment is None:
                    comparison["explanation"] = (
                        f"The VLM did not return an assessment for "
                        f"{matched_item}."
                    )
                else:
                    item_status = assessment.get("status")
                    comparison["vlm_status_or_decision"] = item_status

                    consistency = ppe_result.get(
                        "decision_consistency", {}
                    )
                    ppe_consistency_failed = (
                        isinstance(consistency, dict)
                        and consistency.get("consistent") is False
                    )

                    if ppe_consistency_failed:
                        comparison["reasoning_comparison"] = (
                            "inconclusive"
                        )
                        comparison["explanation"] = (
                            "The VLM reported an internal inconsistency "
                            "in its PPE assessment. Its item-level status "
                            f"for {matched_item} is not treated as reliable "
                            "corroboration or contradiction."
                        )
                    elif item_status == "potentially_missing":
                        comparison["reasoning_comparison"] = (
                            "reasoning_supports"
                        )
                        comparison["explanation"] = (
                            f"The VLM independently assessed "
                            f"{matched_item} as potentially missing."
                        )
                    elif item_status == "present":
                        comparison["reasoning_comparison"] = (
                            "reasoning_conflicts"
                        )
                        comparison["explanation"] = (
                            f"The VLM visually assessed "
                            f"{matched_item} as present. This conflicts "
                            f"with the preliminary detector finding."
                        )
                    else:
                        comparison["reasoning_comparison"] = (
                            "inconclusive"
                        )
                        comparison["explanation"] = (
                            f"The VLM could not establish the status "
                            f"of {matched_item}."
                        )

        elif (
            "potential_worker_machinery_interaction" in category
            or "machinery" in category
            or "equipment_interaction" in category
        ):
            decision = machinery_result.get("decision")
            comparison["vlm_status_or_decision"] = decision

            if decision == "supported":
                comparison["reasoning_comparison"] = (
                    "reasoning_supports"
                )
                comparison["explanation"] = (
                    "The VLM reports direct visual evidence supporting "
                    "a potential worker-machinery interaction."
                )
            elif decision == "not_supported":
                comparison["reasoning_comparison"] = (
                    "reasoning_conflicts"
                )
                comparison["explanation"] = (
                    "The VLM did not find sufficient visual evidence "
                    "to support the potential interaction."
                )
            else:
                comparison["reasoning_comparison"] = "inconclusive"
                comparison["explanation"] = (
                    "The VLM interaction assessment is uncertain."
                )
        else:
            comparison["explanation"] = (
                "No matching VLM assessment category is defined "
                "for this preliminary finding."
            )

        comparisons.append(comparison)

    counts = {
        "reasoning_supports": 0,
        "reasoning_conflicts": 0,
        "inconclusive": 0,
        "not_assessed": 0,
    }

    for comparison in comparisons:
        label = comparison["reasoning_comparison"]
        if label in counts:
            counts[label] += 1

    screening = report.get("screening_outcome", {})
    if not isinstance(screening, dict):
        screening = {}

    return {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": vlm_result.get("model_label"),
        "model_name": vlm_result.get("model_name"),
        "prompt_version": vlm_result.get("prompt_version"),
        "preliminary_screening_outcome": screening.get("outcome"),
        "preliminary_finding_count": len(comparisons),
        "comparisons": comparisons,
        "summary": counts,
        "automatic_alerts_preserved": True,
        "human_review_required_to_complete_analysis": False,
        "interpretation_limit": (
            "This assessment compares fallible detector and rule-based "
            "screening findings with visual reasoning. Agreement does "
            "not establish a confirmed hazard or violation. Disagreement "
            "does not remove or suppress an automatically generated alert."
        ),
    }

