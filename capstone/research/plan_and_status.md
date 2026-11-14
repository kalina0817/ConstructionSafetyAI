# Research plan and implementation status

## Research identity

**Working title:** Context-Aware Vision Reasoning for Proactive Construction
Workplace Safety  
**Degree:** Master of Science in Computer Science capstone  
**Aim:** Design and experimentally evaluate an image-based screening pipeline
that links detected evidence to image-space relationships, explicit rules,
uncertainty, and prevention guidance.

Software implementation is not experimental evidence. “Implemented” below
means code exists; it does not mean an improvement or research contribution
has been demonstrated.

## Research questions

| RQ | Question | Current status | Evidence required to answer |
|---|---|---|---|
| RQ1 | How accurately can computer vision identify relevant objects and extract relationships? | Existing PPE, toolset, and vehicle detectors; reproducible validation runner; scene-graph relationships implemented. | Reproduced per-class detector metrics and an annotated relationship set with a written endpoint-matching protocol. |
| RQ2 | Does spatial, graph, and contextual reasoning improve screening over simpler approaches? | Two rule-triggered categories implemented; independent integrated labels do not yet exist. | Paired detector-only, basic-rule, and graph-linked outputs on a frozen, independent scenario set. |
| RQ3 | Are explanations and recommendations evidence-grounded and uncertainty-aware? | Rules emit detections, boxes, rule IDs, limitations, and uncertainty; worker-level recommendation code is a prototype. | Blinded reviewer rubric; rater-level records, disagreements, unsupported-claim counts, and recommendation safety/relevance scores. |
| RQ4 | What does each reasoning component contribute? | Conditional in the proposal; no ablation results exist. | Implement controlled variants only if the independent labels and project time support a meaningful paired analysis. |

## Implemented scope and exclusions

### Current prototype

- Independent PPE, toolset, and vehicle detector predictions.
- Bounding-box containment and image-space person/machinery edge-distance
  relationships with evidence endpoints.
- Potential missing-PPE and person/machinery interaction screening findings.
- Uncertainty, interpretation limits, evidence traces, and human verification.
- Image upload and structured JSON report; independent scenario evaluator.

### Not assessed

Fall protection, guardrails, elevated edges, excavation protection, exposed
rebar, task requirements, operating machinery state, physical distance, site
exclusion zones, regulatory violations, and injury probability.

## Status by proposal phase

| Phase | Status | Completion gate |
|---|---|---|
| 1 · Research foundation | Proposal supplied; a preliminary targeted literature review is drafted. It is not systematic, and no novelty claim or final research questions have advisor approval. | Expand and verify the related-work review; obtain advisor approval of the gap and final research questions. |
| 2 · Dataset audit | Prepared datasets and labels exist; provenance, public-use rights, annotation mapping, and grouped splits need sign-off. | Dataset manifest, source/license record, class mapping, duplicate grouping, and audit results. |
| 3 · Perception baseline | Vehicle detector: corrected-dataset training and held-out test evaluation completed. PPE/toolset final-test status remains to be confirmed; weak PPE classes still need review. | Review remaining detector baselines, preserve the vehicle test result without tuning against it, and complete the remaining preregistered evaluations. |
| 4 · Contextual reasoning | Initial graph and rules exist with unit tests; empirical relationship/hazard quality is unknown. | Definitions and parameters frozen on validation scenarios; independent error analysis. |
| 5 · Prevention and application | Prototype exists; user effectiveness and recommendation quality are not evaluated. | Evidence-grounded blinded review using the protocol rubric. |
| 6 · Experimental evaluation | Evaluator and protocol exist; independent scenario labels are pending. | Full scenario annotations, rater agreement, paired comparisons, and justified ablations. |
| 7 · Documentation and demonstration | Thesis outline and setup instructions exist. | Complete results, methods, limitations, reproducibility record, presentation, and demo. |


## Completed vehicle detector experiment

**Status:** Training and held-out test evaluation completed for the corrected vehicle dataset. This is a detector experiment, not an evaluation of the integrated hazard-screening system.

### Dataset preparation

- Used a separate corrected copy of the vehicle dataset; the original clean dataset was preserved.
- Corrected five verified Loader images that had been mislabeled as Roller.
- Quarantined three redundant exact-duplicate ForkLift image/label pairs and averaged the retained annotations.
- Applied two precision-only coordinate corrections to keep normalized boxes within image boundaries.
- Structural audit passed with zero malformed labels, invalid classes, invalid coordinates, unreadable images, or missing image/label pairs.
- Final split sizes: 12,454 training images, 2,670 validation images, and 2,670 test images.
- The test split was not modified.

### Training configuration

- Model: YOLO11n, initialized from `yolo11n.pt`.
- Training: 5 epochs, image size 640, batch size 4, CPU, seed 42.
- Training resumed after an interruption and completed. It was not an uninterrupted run.
- Selected checkpoint for final evaluation: `best.pt`.

### Held-out test results

The final evaluation used the corrected dataset's test split of 2,670 images.

| Metric | Result |
|---|---:|
| Precision | 92.31% |
| Recall | 85.17% |
| mAP@50 | 91.74% |
| mAP@50–95 | 75.52% |

Per-class results showed lower mAP@50–95 for Mobile Crane (61.2%) and lower recall for Dump Truck (74.4%). These are descriptive test findings only; they must not be used to tune the model, thresholds, or training procedure.

### Reproducibility artifacts

- Experiment report: `capstone/research/vehicle_corrected_dataset_experiment_report.md`
- Training run: `runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep`
- Final test evaluation: `runs/detect/runs/detect/vehicle_corrected_dataset_final_test_evaluation`
- Corrected dataset: `data/processed/construction-vehicle-detection-clean-loader-corrected`

The preserved experiment report contains the training configuration, validation history, per-class test results, and evaluation artifact inventory. The vehicle test result does not establish hazard-screening accuracy, physical proximity, or real-world safety effectiveness.
## Experimental commitments

1. Keep training, validation, and test roles separate. Tune only with a
   separate validation set. Do not select or tune from test results.
2. Keep related images, video sequences, and perceptual duplicates in one
   partition.
3. Use explicit uncertain/unassessable labels. A missing model output is not
   evidence that PPE is absent.
4. Compare methods on the same images, detector outputs, labels, and operating
   settings.
5. Separate object detection, relationship extraction, binary screening,
   explanation quality, recommendation quality, and efficiency results.
6. Describe the 0.05 normalized machinery edge distance and 0.80 containment
   values as exploratory geometry parameters, not risk measures or calibrated
   physical distances.
7. Treat the vehicle detector's 0.40 confidence default as a provisional
   filter prompted by a manually reviewed false-positive example, not a
   validated operating point; assess any research threshold on validation data.
8. Treat the existing 40-image duplicate audit as a descriptive detector
   review, never as the integrated hazard-screening test set.
9. Compare against ConstructionSite-10k only if tasks, data, annotations, and
   metrics are sufficiently compatible; otherwise describe differences without
   claims of superiority.

## Data rights and reproducibility

The existing PPE dataset metadata lists a private source license. Verify data
and model redistribution rights before public release or inclusion of source
images in the thesis. Keep datasets and weights outside this clean project
folder.

Each experiment should retain the code revision, package versions, hardware,
split, random seed where applicable, dataset/annotation manifest, checkpoint
hashes, parameters, exclusions, report, and evaluator version. Never overwrite
source labels or a locked test set.

## Dataset source and license register

| Dataset | Source | Recorded license | Status |
|---|---|---|---|
| PPE | Roboflow, constructionsafetyai-clean-ann, v1 | Private | Obtain permission before publication or redistribution. |
| Toolset | TariqPlayss Workspace, construction-toolset, v1 | CC BY 4.0 | Provide attribution, license link, and modification notice. |
| Vehicle | jrworkspace, construction-vehicle-detection-8, v3 | CC BY 4.0 | Provide attribution, license link, and modification notice. |

The public dataset licenses do not automatically resolve third-party image rights. Confirm image-level rights and thesis publication requirements with the advisor. Do not redistribute private PPE data or publish its source images until permission is confirmed.
## Advisor decisions to document

- Final novelty claim and approved RQs after literature review.
- Feasible hazard categories and precise positive/negative/uncertain labels.
- Dataset licensing, image publication, and data-retention requirements.
- Annotation sample size, number of independent raters, and timeline.
- Feasibility of RQ4 ablations and external-system comparison.
- Submission format, milestones, and final demonstration requirements.

## Next actions

1. Verify dataset use and publication permissions with the advisor.
2. Expand and verify the preliminary literature review before claiming novelty.
3. Freeze the annotation guide and independent scenario sampling protocol.
4. Collect labels with independent raters before inspecting final test output.
5. Reproduce validation baselines and address known weak PPE classes.
6. Freeze checkpoints/rules, run paired comparisons, and report errors.
7. Write conclusions only from preserved evaluation artifacts.




