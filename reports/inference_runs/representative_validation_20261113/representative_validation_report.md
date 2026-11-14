# Representative Unified Inference Validation

**Date:** November 13, 2026  
**Project:** ConstructionSafetyAI  
**Purpose:** Validate unified inference execution, output generation, dataset-class coverage, and detection-index integrity.

## Scope

A representative sample of 12 images was selected from the PPE, Toolset, and Vehicles validation splits.

| Dataset | Images | Configured classes covered |
|---|---:|---:|
| PPE | 1 | 6 of 11 |
| Toolset | 5 | 35 of 35 |
| Vehicles | 6 | 10 of 10 |
| **Total** | **12** | — |

The PPE sample covered Person and all five `no_*` classes. The Toolset and Vehicles samples covered every configured class in their respective datasets.

Class coverage refers to ground-truth annotations used for sample selection. It does not indicate that the models correctly predicted every class.

## Execution results

- Unified inference completed successfully on all 12 images.
- Each run produced a `detections.json` report and `unified_detections.jpg` visualization.
- All reports contained the expected detection and reasoning arrays.
- No unclassified detections were reported in the scene inventories.
- Person duplicate suppression was observed in two samples.
- Excavator duplicate suppression was observed in two samples:
  - `wShyZ2QrGD6U_jpg.rf.06c6622b5650e743f7606355c2d547c2.jpg` — IoU 0.8518.
  - `yellow-color-excavator-bulldozer-loading-some-rocks-to-red-earth-dump-truck-construction-site-240218603_webp.rf.b0fc82c33e6c08734791869ded266e8e.jpg` — IoU 0.8182.
- Suppressed detections remained represented in raw detector outputs while being excluded from the applicable downstream reasoning results.

## Integrity checks

Automated checks were run against all 12 JSON reports.

| Check | Result |
|---|---:|
| Reports checked | 12 |
| Reports passing | 12 |
| Integrity errors | 0 |
| Raw source indices present and unique | PASS |
| Analysis indices resolve to raw detections | PASS |
| Reasoning indices resolve to analysis detections | PASS |
| Duplicate records reference valid raw detections | PASS |
| Kept detections remain in reasoning output | PASS |
| Suppressed detections excluded from reasoning output | PASS |

## Evidence location

Preserved copies of the run outputs are stored in:

`reports/inference_runs/representative_validation_20261113/`

This directory contains 12 JSON reports and 12 annotated images. Original inference outputs remain in:

`runs/inference/representative_validation/`

## Interpretation and limitations

This validation supports the successful execution and internal consistency of the unified inference pipeline on the selected validation images.

It is not a full-dataset evaluation, an independent human safety assessment, or a measurement of model accuracy. The sample was selected for annotation-class coverage, not statistical representativeness.

The project provides hazard screening rather than safety certification. Image-space proximity does not establish real-world distance or danger. Co-occurrence of a worker and excavator does not by itself establish a hazard, and a missing-PPE prediction does not prove that PPE is absent.

The held-out test split was not used for this validation.

## Regression test results

**Full project test suite — November 13, 2026**

Command: `python -m pytest -v`

- Tests collected: 96
- Tests passed: 96
- Subtests passed: 2
- Failures: 0
- Duration: 80.53 seconds

The full suite covered application behavior, dataset auditing, detector thresholds, inference pipeline configuration, scene-graph reasoning, human review, duplicate handling, and safety workflow evaluation.

Result: PASS.
