"""Worker-level PPE assessment grounded in scene objects and visual evidence."""

import json
from collections import defaultdict

from ppe_safety_rules import RECOMMENDATION_RULES

PPE_ITEMS = {
    "helmet": ({"helmet", "safety helmet"}, "no_helmet"),
    "high_visibility_vest": ({"vest", "safety vest"}, "no_vest"),
    "gloves": ({"gloves"}, "no_gloves"),
    "boots": ({"boots"}, "no_boots"),
    "eye_protection": ({"goggles", "safety glass"}, "no_goggle"),
}


def assess_worker_ppe(graph, visual_rows=None):
    nodes = {node["object_id"]: node for node in graph["nodes"]}
    owners = defaultdict(set)
    for relationship in graph["relationships"]:
        if relationship["predicate"] == "inside_person_bbox":
            owners[relationship["source_id"]].add(relationship["target_id"])

    visual = defaultdict(list)
    if isinstance(visual_rows, list):
        for row in visual_rows:
            if isinstance(row, dict) and isinstance(row.get("worker"), str) and isinstance(row.get("item"), str):
                visual[row["worker"], row["item"]].append(row)

    assessments = []
    for worker in nodes.values():
        if worker["object_category"] != "person":
            continue
        worker_id = worker["object_id"]
        for item, (positive_classes, negative_class) in PPE_ITEMS.items():
            labels = positive_classes | {negative_class.replace("_", " ")}
            evidence = [
                nodes[object_id] for object_id, candidates in owners.items()
                if worker_id in candidates
                and nodes[object_id]["class_name"].strip().lower().replace("_", " ") in labels
            ]
            shared_ids = [node["object_id"] for node in evidence if len(owners[node["object_id"]]) > 1]
            evidence = [node for node in evidence if len(owners[node["object_id"]]) == 1]
            predictions = {
                "potentially_missing" if node["class_name"].strip().lower().startswith("no") else "present"
                for node in evidence
            }
            ambiguous = len(predictions) > 1 or (bool(shared_ids) and not evidence)
            status = "unassessable"
            basis = "No worker-linked detection establishes the item's visibility or status."
            if ambiguous:
                basis = "PPE detections conflict or overlap more than one worker."
            elif visual_rows is None and predictions:
                status = predictions.pop()
                basis = "A detector prediction is inside this worker's box. Wear and visibility are uncertain."
            elif visual_rows is not None:
                candidates = visual[worker_id, item]
                basis = "The selected model did not supply a usable visual assessment for this worker and item."
                if len(candidates) == 1:
                    candidate = candidates[0]
                    visual_basis = candidate.get("visual_basis")
                    if isinstance(visual_basis, str) and visual_basis.strip():
                        basis = visual_basis.strip()
                        claimed = candidate.get("status")
                        if claimed in ("present", "potentially_missing"):
                            if not predictions:
                                basis = "No worker-linked PPE detection corroborates the visual claim. Model observation: " + basis
                            elif claimed not in predictions:
                                basis = "Visual reasoning conflicts with the worker-linked detector prediction. " + basis
                            else:
                                status = claimed
            recommendation = ""
            if status == "potentially_missing":
                recommendation = RECOMMENDATION_RULES[negative_class]["recommendation"]
            elif status == "unassessable":
                recommendation = "Verify this item from a clearer view or on site."
            assessments.append({
                "worker": worker_id,
                "worker_box_xyxy": worker["box_xyxy"],
                "shared_object_ids": shared_ids,
                "item": item,
                "status": status,
                "evidence_object_ids": [node["object_id"] for node in evidence],
                "visual_basis_or_uncertainty": basis,
                "preventive_recommendation": recommendation,
            })
    return assessments


def visual_safety_assessment(graph, assessment):
    """Convert only the selected visual model's grounded observations into alerts."""
    from reasoning_engine import MACHINERY_LIMITATION, MACHINERY_RECOMMENDATION

    analysis = assessment.get("visual_reasoning", {})
    findings = []

    def add_finding(category, object_ids, observation, evidence, recommendation, limitation):
        findings.append({
            "finding_id": f"finding_{len(findings):04d}",
            "finding_category": category,
            "status": "potential",
            "alert_generated_automatically": True,
            "source_object_ids": object_ids,
            "observation": observation,
            "evidence": evidence,
            "recommendation": recommendation,
            "limitation": limitation,
            "uncertainty_reasons": [limitation],
        })

    for row in assessment["worker_ppe"]:
        if row["status"] == "potentially_missing":
            add_finding(
                "potential_missing_ppe",
                [row["worker"], *row["evidence_object_ids"]],
                f"Visual reasoning flagged possible missing {row['item'].replace('_', ' ')}.",
                {"worker_box_xyxy": row["worker_box_xyxy"], "visual_basis": row["visual_basis_or_uncertainty"]},
                row["preventive_recommendation"],
                RECOMMENDATION_RULES[PPE_ITEMS[row["item"]][1]]["limitation"],
            )

    machinery = analysis.get("machinery_assessments", {})
    pairs = []
    for edge in graph["relationships"]:
        if edge["predicate"] != "near_machinery_in_image":
            continue
        judgment = machinery.get(edge["relationship_id"])
        if judgment is None:
            continue
        pairs.append({"worker": edge["source_id"], "machine": edge["target_id"],
                      "relationship_id": edge["relationship_id"], **judgment})
        if judgment["decision"] == "supported":
            add_finding(
                "potential_worker_machinery_interaction",
                [edge["source_id"], edge["target_id"]],
                "Visual reasoning flagged a potential worker–machinery interaction.",
                {"relationship": edge, "visual_evidence": judgment["visual_evidence"]},
                MACHINERY_RECOMMENDATION,
                judgment["limitation"] + " " + MACHINERY_LIMITATION,
            )

    paired_workers = {pair["worker"] for pair in pairs}
    unlinked_workers = [
        row["worker"] for row in analysis.get("worker_context", [])
        if row["subject"] == "person" and row["worker"] not in paired_workers
        and row["position"] not in ("no_identifiable_equipment", "unclear")
    ]
    outcome = "INSUFFICIENT"
    explanation = "The selected model did not provide enough object-linked evidence for a safety alert."
    if findings:
        outcome = "POTENTIAL"
        explanation = f"The selected model generated {len(findings)} potential safety alert(s)."

    elif (assessment["reasoning_status"] == "COMPLETED" and assessment["worker_ppe"]
          and all(row["status"] == "present" for row in assessment["worker_ppe"])
          and not unlinked_workers and all(pair["decision"] == "not_supported" for pair in pairs)):
        outcome = "NO_SUPPORTED"
        explanation = "All five PPE items were assessed for each detected worker. No assessed pair supported an interaction alert."
    return {
        "worker_machinery_assessments": pairs,
        "machinery_coverage": {
            "candidate_pairs": sum(edge["predicate"] == "near_machinery_in_image"
                                   for edge in graph["relationships"]),
            "assessed_pairs": len(pairs),
            "workers_with_equipment_observed_without_candidate_pair": unlinked_workers,
            "limitation": "Machinery outside the detector's candidate pairs has no object-linked interaction assessment.",
        },
        "safety_assessment": {"findings": findings, "status": assessment["reasoning_status"]},
        "screening_outcome": {
            "outcome": outcome,
            "explanation": explanation,
            "finding_count": len(findings),
        },
    }


SCREENING_SYSTEM_PROMPT = (
    "Screen one construction photo for potential PPE and worker-machinery concerns. "
    "Use only visible evidence. Detector boxes are approximate and proximity is in pixels. "
    "Return compact JSON matching the user's structure. Use one short phrase for each observation and limitation."
)


MACHINERY_RELATIONS = ("in_same_scene_as", "on_machine", "handling_machine_or_load", "beside_machine",
                       "other_visible_relation", "no_visible_relation", "unclear")


def parse_visual_screening_output(response):
    result = json.loads(response)
    if not isinstance(result, dict):
        raise ValueError("The screening model must return a JSON object.")
    ppe = result.get("ppe")
    machinery = result.get("machinery")
    if not isinstance(ppe, dict) or not isinstance(machinery, dict):
        raise ValueError("The screening model response must contain PPE and machinery objects.")

    statuses = {"p": "present", "m": "potentially_missing", "u": "unassessable"}
    rows = []
    for worker, entries in ppe.items():
        if not isinstance(entries, list) or len(entries) != len(PPE_ITEMS):
            continue
        for item, entry in zip(PPE_ITEMS, entries):
            if (
                isinstance(entry, list) and len(entry) == 2
                and isinstance(entry[0], str) and entry[0] in statuses
                and isinstance(entry[1], str) and entry[1].strip()
            ):
                rows.append({
                    "worker": worker,
                    "item": item,
                    "status": statuses[entry[0]],
                    "visual_basis": entry[1].strip(),
                })
    for pair, judgment in machinery.items():
        if not isinstance(judgment, dict) or judgment.get("visible_relation") not in MACHINERY_RELATIONS or judgment.get("decision") not in (
            "supported", "not_supported", "unassessable"
        ) or any(not isinstance(judgment.get(k), str) or not judgment[k].strip()
                 for k in ("visual_evidence", "limitation")):
            raise ValueError(f"Invalid evidence or decision for machinery pair {pair}.")
        if (judgment["decision"] == "supported"
                and judgment["visible_relation"] in ("in_same_scene_as", "no_visible_relation", "unclear")):
            judgment["decision"] = "unassessable"
            judgment["limitation"] = "The reported relationship does not establish hazardous exposure. " + judgment["limitation"]

    return {"worker_ppe": rows, "machinery_assessments": machinery}
