# Integrated-system evaluation protocol

## Purpose and claims

Evaluate image-level contextual screening findings against independent
scenario annotations. Do not claim this measures injury risk, legal
compliance, actual machine operation, physical distances, or all hazards at a
site.

## Data separation and locking

- Exclude detector train, validation, and locked test images from contextual
  threshold selection, rule design, and integrated test evaluation.
- Prefer a new scenario-based dataset with explicit provenance, permitted use,
  capture source, inclusion criteria, duplicate groups, and annotation-guide
  version.
- Keep frames from one sequence and near-duplicate images in a single
  partition.
- Freeze labels, model checkpoints, rules, thresholds, and analysis before
  final test evaluation.
- Record every exclusion and any post-test change as exploratory.

## Methods to compare

1. **Detection-only baseline:** report detector outputs without contextual
   screening as an object-level baseline. Do not pretend this alone is a
   hazard classifier.
2. **Basic rule baseline:** apply pre-specified category rules directly to
   detections, without graph relationships.
3. **Graph-linked method:** use object nodes, evidence-linked image-space
   relationships, rules, and explicit uncertainty.
4. **Uncertainty/evidence ablation:** suppress these output components only in
   an offline paired experiment. Do not remove human verification from the
   user-facing workflow.

Use the same frozen detector outputs, examples, annotation guide, and operating
settings across paired system comparisons. Report component differences
without causal language unless the design supports that inference.

## Labels and measures

The evaluator measures the image/category pair for the two supported
categories. Include explicit positive and negative rows; use `uncertain` or
`unassessable` where a label is not defensible. Report excluded counts, support,
TP, FP, FN, TN, precision, recall, specificity, accuracy, and F1 per category
and pooled. Undefined denominators remain null.

Report Cohen's kappa for independent paired binary ratings. Paired records
with uncertain raters are excluded and counted. Retain the original rater
labels and a separate adjudication record.

The image-level evaluator does not measure box localization or relationship
endpoint quality. If making those claims, preregister object identity matching,
IoU criteria, relationship vocabulary, endpoint matching, and treatment of
unmatched or uncertain objects.

## Explanation and recommendation rubric

Blind reviewers to system condition where feasible. Score each item for:

1. Evidence correctness against the source image.
2. Traceability to detection IDs, boxes, relationship, and rule.
3. Separation of observation from inference.
4. Explicit uncertainty and unassessed site conditions.
5. Recommendation relevance, safety, and actionability.
6. Unsupported claims about violations, operation, distance, or calibrated
   risk.

Preserve individual scores and disagreements. Do not combine categories into
one score without a pre-specified aggregation plan.

## Operating thresholds and limitations

The current 0.05 normalized image-space edge distance and 0.80 PPE-box
containment threshold are exploratory geometry parameters. Tune only on
validation scenarios. Do not describe them as risk scores, physical distances,
or empirically selected values until tested independently.

Fall protection, excavation safety, rebar, environment, operation state,
site controls, and regulatory compliance are outside current implementation.
Human review remains required for every screening finding.

## Reproducibility record

Preserve the code revision, dependency versions, hardware, dataset and label
manifest, checkpoint hashes, thresholds, split, random seed when used,
evaluator version, exclusions, raw output reports, metric output, and rubric
records. Never overwrite source data, labels, or locked test reports.
