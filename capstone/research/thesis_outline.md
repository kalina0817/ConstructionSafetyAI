# Master's capstone report outline

## Front matter

- Title page, abstract, acknowledgments, table of contents, lists of figures
  and tables, abbreviations.
- Abstract: problem, research questions, methods, actual results, limitations,
  and conclusion. Do not write results before evaluation.

## 1. Introduction

- Construction-safety screening motivation and decision-support boundary.
- Problem statement, research gap, aim, objectives, and research questions.
- Scope, contributions as hypotheses/targets rather than proven claims.
- Chapter organization.

## 2. Literature review

- Construction-site object detection and PPE monitoring.
- Worker–machinery visual reasoning and spatial relationship extraction.
- Scene graphs and rule-based contextual reasoning.
- Uncertainty, explainability, human factors, and safety decision support.
- Closely related public systems; identify the verified research gap.
- Synthesis and mapping from literature to RQs.

## 3. Data and governance

- Dataset sources, licensing, permitted use, and class mapping.
- Annotation quality, image groups, duplicates, train/validation/test protocol.
- Relationship and scenario labels; annotation guide and rater training.
- Leakage prevention, exclusions, privacy, and reproducible data manifest.

## 4. Methodology

- System architecture and detector baselines.
- Spatial feature and relationship definitions.
- Scene-graph schema, rule library, uncertainty and evidence model.
- Recommendation library and human-review protocol.
- Experimental conditions, metrics, blinding, and analysis plan.

## 5. Implementation

- Reproducible environment, checkpoints, training/evaluation entry points.
- Application, inference, traceability, storage, and error handling.
- Software tests, versioned artifacts, and limitations.

## 6. Results

- Detector metrics with per-class results and uncertainty where estimable.
- Relationship extraction results, only if annotated reference relationships
  and matching criteria exist.
- Paired integrated-system scenario results by category.
- Inter-rater agreement, explanation/recommendation rubric, efficiency.
- Prespecified ablations and sensitivity analyses.

If a measure could not be collected, state that explicitly instead of
substituting an unsupported claim.

## 7. Error analysis and discussion

- False positives, false negatives, uncertain cases, weak classes, and
  representative failures.
- Interpret results against each RQ and related work.
- Explain dataset shift, label uncertainty, external validity, and resource
  limits.
- Separate observed results from interpretation.

## 8. Limitations, ethics, and deployment boundary

- Single-image limits, perspective, occlusion, domain shift, and detector
  errors.
- Data rights and privacy.
- Human verification, safety implications, and prohibited overclaims.
- Conditions required for future deployment or stronger claims.

## 9. Conclusion and future work

- Answer only research questions supported by evidence.
- Summarize defensible contributions and future work.

## Appendices

- Annotation guide, advisor-approved protocol, full metrics, per-class
  analysis, experiment manifest, selected code/configuration, rubric,
  reproducibility instructions, and demonstration materials.
