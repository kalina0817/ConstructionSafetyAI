# Capstone plan-to-implementation register

## Research identity

**Working title:** Context-Aware Vision Reasoning for Proactive Construction
Workplace Safety  
**Degree context:** Master of Science in Computer Science capstone  
**Primary purpose:** Determine whether explicit, image-grounded contextual
reasoning improves construction-image safety screening while preserving
traceability and communicating uncertainty.

This register maps the supplied capstone proposal to the current code and
evidence. “Implemented” means software exists; it does not mean research
effectiveness has been demonstrated.

## Research questions and evidence gates

| Question | Current implementation/evidence | Evidence required before answering |
|---|---|---|
| RQ1: How accurately can the system identify supported objects and extract relevant spatial relationships? | PPE, toolset, and vehicle detector checkpoints; historical training logs and a written detector report. Image-space graph relations are implemented. | Reproduce detector evaluation with recorded hashes and per-class metrics. Define annotated relationship endpoints and a matching protocol before making relationship-extraction claims. |
| RQ2: Does structured contextual reasoning improve safety assessment over simpler detection-based approaches? | Rule-based graph-linked screening is implemented for potential missing-PPE predictions and person/machinery image-space proximity. | Independently label a scenario set; run paired detector-only, basic-rule, and graph-linked conditions on the same frozen examples and rules. Report confidence intervals and category-specific errors where sample size permits. |
| RQ3: Are explanations and preventive recommendations evidence-grounded and uncertainty-aware? | Rules emit source detections, boxes, rule identifiers, uncertainty reasons, limitations, and cautious recommendations. | Conduct blinded rating using a prespecified rubric; retain rater-level scores, disagreements, and unsupported-claim counts. |
| RQ4: What does each reasoning component contribute? | Proposal makes this question conditional; no controlled ablation results exist. | Implement paired ablations and only retain this question if independent annotations and available time support a meaningful comparison. |

## Current scope

### Implemented prototype scope

- Independent PPE, toolset, and vehicle object detector outputs.
- Person duplicate-suppression candidates with a fixed audit report.
- Image-space bounding-box containment and person/machinery edge-distance
  relationships.
- Rule-triggered potential missing-PPE and person/machinery review findings.
- Evidence traceability, explicit uncertainty, limitations, and human review.
- Cautious worker-level recommendations after adjudication.

### Explicitly not assessed

Fall protection, guardrails, excavation protection, exposed rebar, work
activities, machine operating state, true physical distances, site exclusion
zones, regulatory compliance, and injury probability. These remain outside the
implemented validated scope.

## Roadmap and acceptance criteria

| Phase | Current status | Acceptance criteria |
|---|---|---|
| 1. Research foundation | Proposal supplied; literature review and novelty claim not yet evidenced. | Document related work, research gap, final RQs, and advisor-approved scope. |
| 2. Dataset audit | Prepared datasets and validation utilities exist; provenance, licensing, annotation mappings, and cross-split grouping need consolidation. | Record source, permitted use, class mapping, label rules, split/duplicate grouping, and audit outcome for every dataset. |
| 3. Visual-perception baseline | Selected checkpoints, training args, per-epoch results, and a transcribed held-out report exist. Reproduction from a single documented command is pending. | Produce immutable JSON evaluation records with data/checkpoint/config hashes, environment, split, aggregate and per-class metrics. Do not select or tune using test results. |
| 4. Contextual reasoning | Initial graph/rule prototype exists and has focused unit tests; empirical relationship and hazard quality are not established. | Freeze definitions and parameters on validation scenarios; test evidence links, thresholds, uncertainty, and error cases. |
| 5. Prevention and application | Working Streamlit prototype and separate duplicate-review workflow exist. Recommendation quality and user-facing effectiveness are untested. | Preserve source provenance; complete safety, evidence, usability, and blinded recommendation review. |
| 6. Experimental evaluation | Protocol and evaluator exist; no independent scenario test set is available yet. | Collect independent labels for every image/category, retain uncertain cases, complete rater agreement, execute controlled comparisons and prespecified ablations. |
| 7. Final documentation | Setup and evaluation documents exist; thesis results, complete reproducibility record, and presentation remain. | Report methods, actual results, uncertainty, errors, limitations, related work, artifacts, and deviations from proposal. |

## Experimental design commitments

1. Keep training, validation, and test roles distinct. Use validation scenarios
   to tune thresholds; reserve the independent scenario test set for final
   evaluation.
2. Group related images, video frames, and near duplicates in a single
   partition. Record provenance, licensing, inclusion/exclusion, and
   annotation version.
3. Do not treat detector absence as a negative PPE label. Use uncertain or
   unassessable where visibility does not support a judgment.
4. Compare approaches on identical images, checkpoints, annotations, and
   operating settings. The proposed evaluator currently measures image-level
   findings, not localization or relationship endpoint correctness.
5. Keep explanation/recommendation rubric results separate from binary
   hazard-assessment metrics.
6. Describe the 0.05 normalized machinery edge-distance and 0.80 PPE
   containment values as exploratory geometry parameters, not risk measures.
7. Preserve the existing 40-image person duplicate audit as a descriptive
   detector-review set. Do not use it to tune thresholds or claim integrated
   hazard-assessment performance.
8. Treat comparisons with the ConstructionSite-10k implementation as
   conditional on compatible tasks, data, annotations, and measures.

## Dataset and publication considerations

The current PPE `data.yaml` identifies a private Roboflow license. Verify
redistribution and publication rights before including source images,
annotations, or derived material in a public thesis repository. Model weights
and data are local artifacts; source-code publication does not automatically
grant rights to redistribute either.

Before any result is reported, preserve a manifest containing the dataset and
label-guide versions, checkpoint hashes, code revision, dependency versions,
hardware, split, parameters, exclusions, and output report.

## Advisor decisions to record

- Final novelty claim and approved RQs after literature review.
- Final supported hazard categories and definition of each positive label.
- Dataset licensing, image publication, and data-retention permissions.
- Feasible scenario annotation size and number of independent raters.
- Whether RQ4 ablations and the external ConstructionSite-10k comparison are
  feasible under the capstone timeline.
- Final submission format, milestones, and demonstration requirements.

## Immediate sequence

1. Resolve data provenance/licensing and complete the dataset register.
2. Reproduce validation baselines and review existing held-out results without
   changing model selection.
3. Finalize the annotation guide and independent scenario sampling with the
   advisor.
4. Collect labels and assess inter-rater agreement before examining model
   test results.
5. Freeze system versions, run paired experiments, and evaluate explanations.
6. Write results only from the preserved evaluation artifacts.
