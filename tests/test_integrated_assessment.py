import unittest

from vision_language_reasoning import build_integrated_assessment


class IntegratedAssessmentTests(unittest.TestCase):
    def make_report(self):
        return {
            "screening_outcome": {"outcome": "POTENTIAL"},
            "safety_assessment": {
                "findings": [
                    {
                        "finding_id": "finding_ppe",
                        "finding_category": "potential_missing_ppe",
                        "rule_id": "POTENTIAL_MISSING_PPE_PREDICTION",
                        "alert_generated_automatically": True,
                        "evidence": {
                            "detector": "PPE",
                            "class_name": "no_goggle",
                        },
                    },
                    {
                        "finding_id": "finding_machinery",
                        "finding_category": (
                            "potential_worker_machinery_interaction"
                        ),
                        "rule_id": "POTENTIAL_PERSON_MACHINERY_PROXIMITY",
                        "alert_generated_automatically": True,
                        "evidence": {
                            "person_detection": {"class_name": "Person"},
                            "machinery_detection": {
                                "class_name": "Excavator"
                            },
                        },
                    },
                ]
            },
        }

    def make_vlm_result(
        self,
        ppe_status="present",
        ppe_consistent=False,
        machinery_decision="not_supported",
    ):
        return {
            "status": "COMPLETED",
            "model_name": "test-model",
            "analysis": {
                "potential_missing_ppe": {
                    "ppe_item_assessments": [
                        {
                            "item": "eye_protection",
                            "status": ppe_status,
                        }
                    ],
                    "decision_consistency": {
                        "consistent": ppe_consistent,
                    },
                },
                "potential_worker_machinery_interaction": {
                    "decision": machinery_decision,
                },
            },
        }

    def test_finding_category_matches_both_finding_types(self):
        result = build_integrated_assessment(
            self.make_report(),
            self.make_vlm_result(),
        )

        comparisons = {
            item["finding_id"]: item
            for item in result["comparisons"]
        }

        self.assertEqual(
            comparisons["finding_ppe"]["category"],
            "potential_missing_ppe",
        )
        self.assertEqual(
            comparisons["finding_ppe"]["matched_ppe_item"],
            "eye_protection",
        )
        self.assertEqual(
            comparisons["finding_ppe"]["reasoning_comparison"],
            "inconclusive",
        )
        self.assertEqual(
            comparisons["finding_machinery"]["category"],
            "potential_worker_machinery_interaction",
        )
        self.assertEqual(
            comparisons["finding_machinery"]["reasoning_comparison"],
            "reasoning_conflicts",
        )
        self.assertEqual(result["summary"]["not_assessed"], 0)

    def test_inconsistent_ppe_assessment_is_inconclusive(self):
        result = build_integrated_assessment(
            self.make_report(),
            self.make_vlm_result(
                ppe_status="present",
                ppe_consistent=False,
            ),
        )

        ppe = result["comparisons"][0]
        self.assertEqual(
            ppe["reasoning_comparison"],
            "inconclusive",
        )

    def test_consistent_present_ppe_is_a_conflict(self):
        result = build_integrated_assessment(
            self.make_report(),
            self.make_vlm_result(
                ppe_status="present",
                ppe_consistent=True,
            ),
        )

        self.assertEqual(
            result["comparisons"][0]["reasoning_comparison"],
            "reasoning_conflicts",
        )

    def test_consistent_potentially_missing_ppe_supports_finding(self):
        result = build_integrated_assessment(
            self.make_report(),
            self.make_vlm_result(
                ppe_status="potentially_missing",
                ppe_consistent=True,
            ),
        )

        self.assertEqual(
            result["comparisons"][0]["reasoning_comparison"],
            "reasoning_supports",
        )

    def test_automatic_alerts_are_preserved(self):
        result = build_integrated_assessment(
            self.make_report(),
            self.make_vlm_result(),
        )

        self.assertTrue(result["automatic_alerts_preserved"])
        self.assertTrue(
            all(
                item["automatic_alert_preserved"]
                for item in result["comparisons"]
            )
        )


if __name__ == "__main__":
    unittest.main()


def test_visual_ppe_claim_needs_matching_worker_evidence():
    from screening_assessment import assess_worker_ppe

    graph = {
        "nodes": [
            {"object_id": "w", "object_category": "person", "box_xyxy": [0, 0, 60, 100]},
            {"object_id": "h", "object_category": "ppe_prediction", "class_name": "helmet"},
        ],
        "relationships": [{"source_id": "h", "target_id": "w", "predicate": "inside_person_bbox"}],
    }
    claims = [
        {"worker": "w", "item": item, "status": status, "visual_basis": "Model observation."}
        for item, status in [("helmet", "present"), ("boots", "present"), ("gloves", "potentially_missing")]
    ]
    rows = {row["item"]: row for row in assess_worker_ppe(graph, claims)}
    assert rows["helmet"]["status"] == "present"
    for item in ("boots", "gloves"):
        assert rows[item]["status"] == "unassessable"
        assert rows[item]["evidence_object_ids"] == []


def test_shared_helmet_preserves_independent_worker_evidence():
    from screening_assessment import assess_worker_ppe

    graph = {
        "nodes": [
            {"object_id": worker, "object_category": "person", "box_xyxy": [0, 0, 60, 100]}
            for worker in ("left", "right")
        ] + [
            {"object_id": item, "object_category": "ppe_prediction", "class_name": "helmet"}
            for item in ("shared", "exclusive")
        ],
        "relationships": [
            {"source_id": item, "predicate": "inside_person_bbox", "target_id": worker}
            for item, worker in [("shared", "left"), ("shared", "right"), ("exclusive", "right")]
        ],
    }
    claims = [{"worker": w, "item": "helmet", "status": "present", "visual_basis": "Helmet visible."}
              for w in ("left", "right")]
    for visual_rows in (None, claims):
        helmets = {r["worker"]: r for r in assess_worker_ppe(graph, visual_rows) if r["item"] == "helmet"}
        assert helmets["left"]["status"] == "unassessable"
        assert helmets["left"]["evidence_object_ids"] == []
        assert helmets["right"]["status"] == "present"
        assert helmets["right"]["evidence_object_ids"] == ["exclusive"]
        assert all(r["shared_object_ids"] == ["shared"] for r in helmets.values())



def test_screening_rules_group_missing_ppe_by_worker_and_item():
    from reasoning_engine import evaluate_safety_rules
    from screening_assessment import assess_worker_ppe

    workers = [{"object_id": w, "object_category": "person", "class_name": "Person",
                "box_xyxy": [0, 0, 80, 100]} for w in ("left", "right")]
    ppe = [{"object_id": i, "object_category": "missing_ppe_prediction",
            "class_name": "no_gloves", "confidence": 0.8, "detector": "PPE",
            "box_xyxy": [10, 30, 18, 40]} for i in ("g1", "g2", "shared", "orphan")]
    graph = {"nodes": workers + ppe, "relationships": [
        {"source_id": i, "target_id": w, "predicate": "inside_person_bbox"}
        for i, w in (("g1", "left"), ("g2", "left"), ("shared", "left"), ("shared", "right"))
    ]}
    rows = assess_worker_ppe(graph)
    findings = evaluate_safety_rules(graph, worker_ppe=rows)["findings"]
    assert [f["source_object_ids"] for f in findings] == [["left", "g1", "g2"]]
    assert len(evaluate_safety_rules(graph)["findings"]) == 4


def test_visual_pair_decisions_generate_only_supported_alerts():
    from screening_assessment import visual_safety_assessment

    decisions = ["supported", "not_supported", "unassessable"]
    pairs = {d: {"decision": d, "visual_evidence": "Pair-specific observation",
                 "limitation": "Depth is unknown."} for d in decisions}
    graph = {"relationships": [{"relationship_id": d, "source_id": "worker",
              "target_id": "machine_" + d, "predicate": "near_machinery_in_image"}
              for d in decisions]}
    result = visual_safety_assessment(graph, {"worker_ppe": [], "reasoning_status": "COMPLETED",
                                      "visual_reasoning": {"machinery_assessments": pairs}})
    assert len(result["worker_machinery_assessments"]) == 3
    findings = result["safety_assessment"]["findings"]
    assert [f["source_object_ids"] for f in findings] == [["worker", "machine_supported"]]



def test_streamed_visual_reasoning_requires_a_complete_response():
    import json
    from unittest.mock import MagicMock, patch
    import pytest
    from vision_language_reasoning import call_model

    response = MagicMock()
    response.__enter__.return_value = response
    first = {"message": {"content": "one"}, "done": False}
    last = {"message": {"content": " two"}, "done": True, "eval_count": 2}
    with patch("vision_language_reasoning.ensure_local_ollama"), patch(
        "vision_language_reasoning.requests.post", return_value=response
    ) as post:
        response.iter_lines.return_value = [json.dumps(first), json.dumps(last)]
        text, metadata = call_model("model", "image", "prompt", screening=True)
        assert text == "one two"
        assert metadata["eval_count"] == 2
        assert post.call_args.kwargs["stream"]
        response.iter_lines.return_value = [json.dumps(first)]
        with pytest.raises(RuntimeError, match="before model finished"):
            call_model("model", "image", "prompt", screening=True)



def test_unsupported_object_relation_cannot_ground_a_hazard():
    import json
    from screening_assessment import parse_visual_screening_output

    raw = {"ppe": {}, "machinery": {"edge": {"visible_relation": "unclear",
           "decision": "supported", "visual_evidence": "Object partly hidden",
           "limitation": "Identity is uncertain."}}}
    analysis = parse_visual_screening_output(json.dumps(raw))
    assert analysis["machinery_assessments"]["edge"]["decision"] == "unassessable"



def test_close_up_identifies_headwear_and_separates_position_from_exposure():
    import json
    from screening_assessment import PPE_ITEMS
    from vision_language_reasoning import read_worker_observations

    response = {"subject": "person", "headwear": {"appearance": "soft_cap", "reason": "Soft fabric brim"},
                "ppe": {name: {"status": "u", "reason": "Obscured"} for name in PPE_ITEMS if name != "helmet"},
                "equipment_context": {"machine_identity": "construction_vehicle", "body_support": "seated_in_operator_station", "position": "seated_on_equipment", "reason": "Seat visible"},
                "machinery": {"e1": {"machine_identity": "construction_vehicle", "body_support": "seated_in_operator_station", "position": "seated_on_equipment", "decision": "not_supported",
                                     "visual_evidence": "Seated in cab", "limitation": "Motion unknown"}}}
    rows, context = read_worker_observations(json.dumps(response), "w", ["e1"])
    assert rows["worker_ppe"][0]["status"] == "potentially_missing"
    assert rows["machinery_assessments"]["e1"]["visible_relation"] == "on_machine"
    assert rows["machinery_assessments"]["e1"]["decision"] == "not_supported"
    assert context["worker"] == "w"
    response["machinery"]["e1"]["machine_identity"] = "ordinary_surface_or_hand_tool"
    response["machinery"]["e1"]["decision"] = "supported"
    rows, context = read_worker_observations(json.dumps(response), "w", ["e1"])
    assert rows["machinery_assessments"]["e1"]["visible_relation"] == "no_visible_relation"
    assert rows["machinery_assessments"]["e1"]["decision"] == "not_supported"
    response["subject"] = "not_person"
    rows, context = read_worker_observations(json.dumps(response), "w", ["e1"])
    assert all(row["status"] == "unassessable" for row in rows["worker_ppe"])
    assert rows["machinery_assessments"]["e1"]["decision"] == "unassessable"
    assert context["position"] == "unclear"
    assert "Seated in cab" not in rows["machinery_assessments"]["e1"]["visual_evidence"]


def test_equipment_seen_without_a_pair_is_a_coverage_gap():
    from screening_assessment import visual_safety_assessment

    result = visual_safety_assessment({"relationships": []}, {
        "reasoning_status": "COMPLETED", "worker_ppe": [{"status": "present"}],
        "visual_reasoning": {"worker_context": [{"worker": "w", "subject": "person",
                                                  "position": "standing_on_equipment"}]}})
    assert result["machinery_coverage"]["workers_with_equipment_observed_without_candidate_pair"] == ["w"]
    assert result["screening_outcome"]["outcome"] == "INSUFFICIENT"
    assert result["worker_machinery_assessments"] == []
    assert result["safety_assessment"]["findings"] == []



def test_rejected_crop_with_two_unique_ppe_classes_rechecks_same_model(tmp_path):
    import json
    from unittest.mock import patch
    from PIL import Image
    from screening_assessment import PPE_ITEMS
    from vision_language_reasoning import screen_worker_views

    path = tmp_path / "photo.jpg"
    Image.new("RGB", (32, 32)).save(path)
    graph = {"nodes": [{"object_id": "w", "object_category": "person", "box_xyxy": [1, 1, 30, 30]}]
             + [{"object_id": name, "object_category": "ppe_prediction", "class_name": name}
                for name in ("helmet", "vest")],
             "relationships": [{"predicate": "inside_person_bbox", "source_id": name, "target_id": "w"}
                               for name in ("helmet", "vest")]}
    response = {"headwear": {"appearance": "rigid_hard_hat", "reason": "Rigid shell visible"},
                "ppe": {name: {"status": "u", "reason": "Obscured"} for name in PPE_ITEMS if name != "helmet"},
                "equipment_context": {"machine_identity": "no_machine_visible",
                                      "position": "no_identifiable_equipment", "reason": "No machine visible"},
                "machinery": {}}
    results = [(json.dumps({**response, "subject": subject}), {"elapsed_seconds": 1})
               for subject in ("unclear", "person")]
    record = {}
    with patch("vision_language_reasoning.call_model", side_effect=results) as model:
        screen_worker_views(path, graph, "selected-model", record)
    assert [call.args[0] for call in model.call_args_list] == ["selected-model"] * 2
    assert model.call_args_list[0].args[1] != model.call_args_list[1].args[1]
    assert record["analysis"]["worker_ppe"][0]["status"] == "present"
    assert set(record["raw_response"]) == {"w", "w_original_view"}



def test_head_detail_requires_agreement_with_worker_linked_evidence(tmp_path):
    import json
    from unittest.mock import patch
    from PIL import Image
    from screening_assessment import PPE_ITEMS
    from vision_language_reasoning import screen_worker_views

    image = tmp_path / "worker.jpg"
    Image.new("RGB", (64, 64)).save(image)
    graph = {"nodes": [{"object_id": "w", "object_category": "person", "box_xyxy": [0, 0, 50, 60]},
                       {"object_id": "h", "class_name": "helmet", "object_category": "ppe_prediction",
                        "box_xyxy": [5, 3, 18, 20]}],
             "relationships": [{"source_id": "h", "target_id": "w", "predicate": "inside_person_bbox"}]}
    first = {"subject": "person", "headwear": {"appearance": "soft_cap", "reason": "Brim visible"},
             "ppe": {name: {"status": "u", "reason": "Obscured"} for name in PPE_ITEMS if name != "helmet"},
             "equipment_context": {"machine_identity": "no_machine_visible", "position": "no_identifiable_equipment",
                                   "reason": "No machine"}, "machinery": {}}
    for appearance, expected in (("rigid_hard_hat", "present"), ("soft_cap", "unassessable")):
        detail = {"appearance": appearance, "reason": "Closer view of head"}
        with patch("vision_language_reasoning.call_model", side_effect=[
            (json.dumps(value), {"elapsed_seconds": 1}) for value in (first, detail)
        ]) as model:
            record = {}
            screen_worker_views(image, graph, "chosen", record)
        assert record["analysis"]["worker_ppe"][0]["status"] == expected
        assert list(record["raw_response"]) == ["w", "w_head_detail"]
        assert all(call.args[0] == "chosen" for call in model.call_args_list)



def test_on_machine_requires_matching_body_support():
    from vision_language_reasoning import grounded_position
    for position, supporting in (("standing_on_equipment", "feet_on_machine_step_or_deck"),
                                 ("seated_on_equipment", "seated_in_operator_station")):
        observation = {"machine_identity": "construction_vehicle", "position": position}
        for support in ("not_visible", "independent_surface"):
            assert grounded_position({**observation, "body_support": support}) == "in_same_scene_as"
        assert grounded_position({**observation, "body_support": supporting}) == position
        assert grounded_position({**observation, "position": "in_same_scene_as", "body_support": supporting}) == position
