# Integrated-system evaluation protocol

## Purpose and scope

This protocol evaluates the image-level screening findings produced by the contextual reasoning layer. It does not claim to evaluate real-world injury risk, regulatory compliance, or physical worker-to-machine distances.

The current implemented rule categories are:

- `potential_missing_ppe`
- `potential_worker_machinery_interaction`

Fall protection, excavations, rebar exposure, operating status, and site exclusion-zone compliance are not currently assessed by the system.

The current scene-graph implementation uses a normalized image-space bounding-box edge-distance threshold of `0.05` for machinery review candidates and a `0.80` PPE-box containment threshold. These are exploratory geometric settings, not empirically calibrated values. Freeze or tune them on a separate validation set before evaluating against the independent scenario test set.

## Dataset separation

Create a separate scenario-based image set for contextual evaluation. Do not reuse detector training, validation, or locked test images for rule tuning, threshold selection, or recommendation design.

Record image provenance, permitted use, inclusion/exclusion criteria, capture source, and duplicate grouping. If scenarios are related or originate from the same sequence, keep the whole group in one partition.

Freeze the labels and system parameters before final evaluation. Any changes made after seeing test results require a new independent test set or must be reported as exploratory.

## Annotation unit and CSV schema

The annotation unit is one `(image_id, finding_category)` pair. Label every included image for every evaluated category, including explicit negatives. Omitting a row means "not evaluated," not "no hazard."

Required columns:

| Column | Meaning |
|---|---|
| `image_id` | Unique source image filename |
| `finding_category` | One supported category |
| `ground_truth` | `1` for positive, `0` for negative, or `uncertain` / `unassessable` |

An optional `evidence_note` column may record visible evidence or explain uncertainty. It is not required by the evaluator and is not scored.

This evaluation uses one human annotator. The annotator must review the source images independently before viewing model predictions. Labels must be based on the written annotation guide, not on whether the system generated a finding.

Preserve the original human labels. Do not change labels to agree with model predictions. Use `uncertain` or `unassessable` rather than guessing when the image does not support a confident decision.

The evaluator requires only the three columns listed above. Second-rater columns, adjudication, inter-rater agreement, and Cohen's kappa are not applicable to this single-annotator evaluation.

Ground truth describes the visual screening condition defined by the annotation guide. It must not assert that a regulatory violation or actual incident occurred based only on a photograph.

## Recommended label definitions

### Potential missing-PPE screening condition

A positive label requires that the target worker and relevant PPE category can be assessed in the image and that the annotation guide identifies the equipment as visibly absent.

A negative label requires that the relevant body region is sufficiently visible and assessable and that the supported PPE items being evaluated appear present.

Mark occluded, too-small, ambiguous, or otherwise unassessable cases as `uncertain` or `unassessable`. The evaluator excludes these pairs from binary metrics while retaining and reporting their counts.

Do not equate absence of a detector output with absence of PPE. A missing detection alone is not evidence that equipment is absent.

### Potential worker-machinery interaction screening condition

Use the human visual annotation criterion defined in the annotation guide. Compare the worker bounding box with visible machinery bounding boxes:

- **Positive:** The boxes overlap, touch, or have an edge-to-edge gap no greater than one worker bounding-box height (`y_max - y_min`).
- **Negative:** The image is assessable and every visible machinery box is farther than one worker bounding-box height from the worker box.
- **Uncertain:** Perspective, occlusion, ambiguous object identity, or unclear box boundaries prevent a confident decision.
- **Unassessable:** The worker or machinery is not sufficiently visible to make the comparison.

This is an image-space screening criterion. It does not establish physical distance, operating status, actual danger, or an incident.

### Difference between human labels and system screening

The human machinery annotation criterion and the implemented system threshold are separate definitions and must not be treated as equivalent.

The scene-graph system uses normalized image-space edge distance:

`edge_distance_pixels / image_diagonal <= 0.05`

The human annotation criterion uses a gap relative to worker bounding-box height. Differences between these criteria may contribute to false positives or false negatives and should be considered when interpreting results.

The system's `0.80` PPE-box containment threshold is also an exploratory geometric setting, not an empirically calibrated measure of PPE absence.

## Measures

`evaluate_safety_system.py` reports per-category and pooled true positives, false positives, true negatives, false negatives, precision, recall, specificity, accuracy, and F1.

Report positive and negative support alongside all scores. Scores are undefined when their denominator is zero. Report the number of uncertain and unassessable pairs separately, including the number excluded from binary metrics.

Report the results as a single-annotator image-level evaluation. Do not report inter-rater reliability or Cohen's kappa.

For localization or relationship-extraction claims, create a separately defined object/relationship matching protocol and evaluate the evidence coordinates and relationship endpoints against annotations. The current image-level evaluator does not measure box localization or edge matching.

## Explanation and recommendation review

Review a stratified sample of outputs using a predefined rubric. Where feasible, score explanations without first viewing the corresponding scenario ground-truth labels to reduce confirmation bias.

Score each item for:

1. Evidence correctness and whether cited detections exist in the source image.
2. Traceability to detection IDs, bounding boxes, and a named rule.
3. Whether the explanation distinguishes observation from inference.
4. Whether uncertainty and unassessed site conditions are explicit.
5. Recommendation relevance, safety, and actionability.
6. Unsupported claims, including false statements of violation, operating status, calibrated risk, physical distance, or regulatory requirements.

Retain individual item scores and supporting notes. Report this as a single-reviewer qualitative assessment, not as inter-reviewer agreement. Do not collapse the qualitative rubric into a single accuracy score without a pre-specified aggregation method.

## Ablations

Run the same frozen scenario set and annotation protocol for:

1. Detector-only candidates.
2. Basic detector plus explicit rule screening.
3. Full graph-linked contextual screening.
4. Full system with uncertainty and evidence displays suppressed, for an offline analytical ablation only.

Do not present an ablation as causal evidence unless the input data, model checkpoints, thresholds, and evaluation examples are otherwise held constant.

The current version records graph evidence and uncertainty; ablation variants and their outputs still need to be implemented and evaluated.

## Reproducibility record

For each evaluation, preserve the code revision, dependency versions, model checkpoint hashes, dataset manifest, label-guide revision, threshold settings, hardware, output reports, metric script version, and any exclusions.

Never overwrite source annotations or the locked detector test sets.