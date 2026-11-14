# Scenario Annotation Guide

## Purpose and annotation unit

This guide defines the human reference labels for image-level safety screening evaluation.

One record represents one `(image_id, finding_category)` pair. Annotate every included image for every evaluated category. An omitted pair is not a negative.

Supported categories:

- `potential_missing_ppe`
- `potential_worker_machinery_interaction`

Allowed labels:

- `1` / `positive`
- `0` / `negative`
- `uncertain`
- `unassessable`

The project has one human annotator. The `ground_truth` column records that annotator's final independent visual judgment. Do not use model predictions to determine reference labels.

## General annotation procedure

1. Confirm that the image belongs to the approved evaluation set and can be opened.
2. Review the original image without viewing the system prediction or its findings.
3. Apply the category-specific definitions below.
4. Record the label and a concise evidence note describing visible evidence.
5. Use `uncertain` or `unassessable` instead of guessing.
6. Preserve the original annotation records. If a label is corrected, document the change and its reason.
7. Never label a photograph as proof of an actual incident, regulatory violation, or unsafe operating condition.

## Potential missing-PPE screening condition

This category concerns visible evidence that one or more supported PPE items appear absent.

**Positive:** At least one supported PPE item appears visibly absent from its relevant body region, and that region is sufficiently visible to make the judgment. This is a visual screening label; it does not establish that the item was legally or operationally required.

**Negative:** The relevant visible body regions can be assessed, and the supported PPE items being evaluated appear present.

**Uncertain:** The image provides some relevant evidence, but visibility, identity, occlusion, or ambiguity prevents a confident positive or negative judgment.

**Unassessable:** The relevant worker or body region is not sufficiently visible to evaluate the condition.

Do not equate a missing detector prediction with visibly absent PPE. Do not infer task-specific PPE requirements without independently verified task information. A negative label does not certify PPE compliance.

## Potential workerâ€“machinery interaction screening condition

This category uses a defined image-space proximity criterion. It does not measure real-world physical distance.

For human annotation, compare the visible worker bounding box with the visible machinery bounding box. Let the worker bounding-box height be:

`worker_box_height = y_max - y_min`

Measure the shortest Euclidean edge-to-edge gap between the two boxes.

**Positive:** The worker and machinery boxes overlap, touch, or have an edge-to-edge gap no greater than one worker bounding-box height.

**Negative:** The image is assessable, and every visible machinery box is farther than one worker-box height from the worker box.

**Uncertain:** The worker and machinery are identifiable, but perspective, occlusion, unclear box boundaries, or other ambiguity prevents a defensible proximity judgment.

**Unassessable:** The worker or machinery is not sufficiently visible to make the comparison.

This is an image-space annotation rule only. It does not establish physical separation, machine movement, operating status, an exclusion-zone boundary, or actual danger. A positive label means the visual arrangement warrants verification, not that a hazard or incident is confirmed.

## Difference between human labels and system screening

The human annotation criterion above is not the same as the implemented system threshold.

The system calculates Euclidean edge-to-edge distance between the person and machinery boxes and divides it by the image diagonal:

`normalized_distance = edge_distance / image_diagonal`

The current system creates a `near_machinery_in_image` relationship when this value is at most `0.05`.

The human reference criterion uses one worker-box height instead. Keep these definitions distinct when reporting evaluation results. Do not change the system threshold using the final evaluation set.

System-generated findings are screening candidates requiring human verification. They are not reference labels.

## Independent annotation and evaluation safeguards

- Complete the human annotations before reviewing model predictions.
- Do not revise a human label merely to agree with a system finding.
- Keep evidence notes grounded in what is visible in the source image.
- Freeze the annotation guide and evaluation labels before final test evaluation.
- Do not use the final evaluation set to tune system thresholds or rules.
- Record any later changes to the guide, labels, or implementation. Results after such changes must be identified as exploratory unless evaluated on a separate untouched test set.

## CSV schema

The evaluator requires these three columns:

```csv
image_id,finding_category,ground_truth
scenario_001.jpg,potential_missing_ppe,uncertain
scenario_001.jpg,potential_worker_machinery_interaction,positive
scenario_002.jpg,potential_missing_ppe,negative
scenario_002.jpg,potential_worker_machinery_interaction,unassessable
```

An optional `evidence_note` column may be included for annotation documentation. It is not used in metric calculations.

The evaluator accepts the required three-column schema without `rater_1` or `rater_2`. Uncertain and unassessable labels are excluded from binary metric calculations and counted separately in the evaluation summary.

There is one human annotator; inter-rater agreement and Cohen's kappa are therefore not applicable.





