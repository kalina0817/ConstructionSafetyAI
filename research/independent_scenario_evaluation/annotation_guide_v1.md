# Independent Scenario Annotation Guide — Version 1.0

## Purpose

This guide defines single-annotator, image-level labels for the independent scenario evaluation of contextual construction-site safety screening.

Labels describe visible image evidence only. They do not establish regulatory violations, actual danger, physical distance, machine operating status, or incidents.

## Annotation procedure

1. Review each original candidate image without opening model predictions, detection overlays, or system reports.
2. Annotate both supported categories for every included image.
3. Record one label per `(image_id, finding_category)` pair.
4. Add concise visual evidence or uncertainty explanations in `evidence_note` when useful.
5. Preserve the original labels. Do not revise them to agree with model predictions.

## Allowed labels

- `1`: Positive; the defined visual screening condition is supported.
- `0`: Negative; the image is sufficiently assessable and the condition is not supported.
- `uncertain`: Relevant evidence is visible but ambiguous or conflicting.
- `unassessable`: Required people, body regions, or objects cannot be assessed adequately.

Do not guess. Use `uncertain` or `unassessable` when evidence is insufficient.

## Category 1: potential_missing_ppe

Evaluate five PPE categories:

| Human-facing item | Standardized detector class names |
|---|---|
| Safety boots | `boots`, `no_boots` |
| Gloves | `gloves`, `no_gloves` |
| Goggles / eye protection | `goggles`, `no_goggle` |
| Hard hat / helmet | `helmet`, `no_helmet` |
| High-visibility vest | `vest`, `no_vest` |

The standardized dataset also contains `Person`. These detector classes are reference vocabulary only. Detector outputs must not determine human ground truth.

### Positive (`1`)

At least one of the five supported PPE items is clearly visibly absent on an assessable target worker.

If multiple workers are present, a clearly visible absence on at least one assessable worker is sufficient for a positive image-level label.

### Negative (`0`)

The relevant body regions can be assessed and all five evaluated PPE items appear present on the target worker or workers.

### Uncertain (`uncertain`)

The image contains ambiguous or conflicting visual evidence that prevents a confident positive or negative decision.

### Unassessable (`unassessable`)

The worker or relevant body regions are too small, cropped, obscured, or otherwise not visible enough to assess the PPE items.

Do not infer PPE absence from an item being outside the frame. Do not infer that PPE is legally or operationally required. This category measures visible missing-PPE screening conditions, not compliance.

## Category 2: potential_worker_machinery_interaction

Use visible worker and machinery bounding boxes as a consistent image-space comparison.

For each assessable worker, identify visible construction machinery or vehicles and compare the worker box with each machinery box.

Worker box height:

`worker_height = y_max - y_min`

Edge-to-edge gap is the shortest nonnegative distance between the two boxes. Overlapping or touching boxes have a gap of zero.

### Positive (`1`)

At least one visible machinery box overlaps or touches a worker box, or its edge-to-edge gap is no greater than that worker's bounding-box height.

### Negative (`0`)

The image is assessable, and every visible machinery box is farther from the worker box than one worker bounding-box height.

If no machinery is visible, label `0` when the scene is sufficiently assessable to establish that fact.

### Uncertain (`uncertain`)

Perspective, occlusion, ambiguous object identity, or unclear box boundaries prevent a confident comparison.

### Unassessable (`unassessable`)

The worker or relevant machinery is not sufficiently visible to make the comparison.

This is an image-space visual criterion only. It does not establish physical distance, machine operation, actual danger, or an incident.

## Important distinction from system thresholds

The human machinery criterion uses a gap relative to worker bounding-box height.

The system uses:

`edge_distance_pixels / image_diagonal <= 0.05`

These are different criteria and must not be treated as equivalent.

The system's PPE-box containment threshold of `0.80` is also an exploratory geometric setting. Neither system threshold determines human labels.

## CSV requirements

File: `independent_scenario_annotations.csv`

Required columns:

- `image_id`
- `finding_category`
- `ground_truth`

Optional column:

- `evidence_note`

Each included image must have exactly two rows: one for `potential_missing_ppe` and one for `potential_worker_machinery_interaction`.

Do not add second-rater columns. This is a single-annotator evaluation.

## Independence and recordkeeping

- Complete and save human labels before viewing model predictions.
- Do not copy preliminary inventory PPE statuses into the ground-truth CSV.
- Do not use detector outputs to decide whether PPE is present or absent.
- Preserve the original CSV and guide as evaluation artifacts.
- Keep related images or source groups together if a later validation/test split is created.
- Record any exclusions and reasons separately.
