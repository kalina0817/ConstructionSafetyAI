# Construction Safety AI

> **Clean master's capstone workspace:** [capstone/](capstone/README.md).
> New independent research application, methods, reproducibility scripts, and
> tests live there. The existing workspace is preserved while the clean
> project is verified.

Construction Safety AI is a research prototype for screening construction-site
images with PPE, toolset, and vehicle object detectors. It combines independent
model predictions, prepares them for human review, and can generate cautious
worker-level recommendations from findings that a reviewer confirms.

This software is **not** a safety-certification system, regulatory assessment,
or substitute for a qualified safety professional. A detector prediction,
bounding-box overlap, or image-space distance is not proof of a violation,
physical distance, or hazard.

## Current project state

- Raw datasets are stored under `data/raw/`; prepared datasets are under
  `data/processed/`.
- Dataset audits and preparation utilities are in the project root and
  `Scripts/`.
- Training run directories under `runs/detect/` contain model checkpoints,
  arguments, and validation logs.
- The written baseline report at
  `data/reports/ConstructionSafetyAI_Dataset_Training_Evaluation_Report.docx`
  records held-out test metrics, including per-class results for all three
  detectors.
- `unified_inference.py` runs the three selected checkpoints on one image and
  writes its report and annotated image under `reports/inference_runs/`.
- `integrated_app.py` provides image upload, inference, human review, and the
  adjudication-to-recommendation workflow.

Baseline detector training and held-out test evaluation have been completed
and reported. Reproducing those results from code is still incomplete: there
is no checked-in training/evaluation entry point or machine-readable final
test report. The integrated review and recommendation workflow also needs its
own independent evaluation; detector metrics alone do not validate it.

The written report records these held-out baselines:

| Detector | Test images | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| PPE | 141 | 0.654 | 0.676 | 0.681 | 0.507 |
| Toolset | 527 | 0.478 | 0.409 | 0.405 | 0.248 |
| Vehicles | 2,669 | 0.930 | 0.841 | 0.921 | 0.765 |

These are values transcribed from the report, not rerun as part of the current
code changes. The report also shows substantially weaker results for several
missing-PPE and toolset classes.

## Environment setup

Use Python 3.11 and a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The trained checkpoints are expected at:

- `runs/detect/runs/detect/ppe_yolo11n_postcorrection_15ep/weights/best.pt`
- `runs/detect/runs/detect/toolset_yolo11n_continuation_final5-2/weights/best.pt`
- `runs/detect/runs/detect/vehicle_yolo11n_continuation_5ep/weights/best.pt`

Those model files are local artifacts; the dependency file does not download
them.

## Run the application

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The capstone workspace opens to a research overview with navigation to
research methods, image screening, and the person-duplicate audit. On the
screening page, select an image, run inference, review every candidate, and
save the review before generating the adjudicated report and recommendations.
Human decisions are stored separately from detector predictions.

The original single-purpose screening page remains available:

```powershell
streamlit run integrated_app.py
```

For command-line inference:

```powershell
python unified_inference.py --image path\to\image.jpg --conf 0.25 --imgsz 640
```

Reproduce detector validation metrics and preserve checkpoint and dataset
hashes:

```powershell
python evaluate_detector_baselines.py --model ppe --split val
python evaluate_detector_baselines.py --model toolset --split val
python evaluate_detector_baselines.py --model vehicles --split val
```

Validation is the default. The `test` split is locked and requires
`--allow-locked-test`; use that flag only for the preregistered final
evaluation after model selection and threshold tuning are complete. Evaluation
records are saved under `reports/detector_evaluations/`.

To run an individual report stage, specify the exact inference directory:

```powershell
python build_adjudicated_report.py --run-folder reports\inference_runs\<run-id>
python associate_worker_findings.py --run-folder reports\inference_runs\<run-id>
python safety_reasoning.py --run-folder reports\inference_runs\<run-id>
```

The Streamlit app passes the selected run directory to each stage
automatically. Explicit run selection prevents scripts from silently using a
stale report.

## Dataset validation

Validate the corrected PPE dataset's label pairs, class IDs, normalized box
coordinates, and image bounds:

```powershell
python validate_annotations.py
```

Validation errors return a nonzero exit code. Keep backup and review files
outside dataset split folders so dataset tools only see active annotations.

## Automated checks

Run the CPU-only pipeline regression tests:

```powershell
python -m unittest discover -s tests -v
```

## Research work required before thesis evaluation

1. Reproduce the reported baseline training and test evaluation from a
   documented command, recording hyperparameters, seeds, software versions,
   checkpoint provenance, and per-class metrics.
2. Keep the existing held-out test sets locked; do not tune thresholds or
   models against them. Use a separate scenario-based set to assess the
   integrated workflow.
3. Measure human-review agreement and document reviewer instructions,
   adjudication, false positives, and false negatives. The latest 40-image
   duplicate report has 5 manually reviewed images and 35 still awaiting
   review.
4. Investigate false positives and weaker classes reported in the baseline,
   especially missing-PPE classes, toolset gloves/safety glasses, and
   machinery predictions that create spurious spatial candidates.
5. Treat spatial relationships as image-space metadata only; do not present
   them as physical proximity or risk estimates without calibrated evidence
   and domain validation.

## Person duplicate review

The fixed 40-image duplicate-filter audit has a separate reviewer interface:

```powershell
streamlit run review_person_duplicates.py
```

The interface shows each source image and a person-only overlay, distinguishes
retained from suppressed person boxes, records the visible-person count and
per-suppression judgments, and saves annotations to
`reports/person_duplicate_validation_final_40_reviews.json`. Existing
manually reviewed images are excluded from the initial queue. The original
CSV, Markdown report, inference outputs, dataset labels, and splits are not
modified. This is a descriptive audit set; do not use it to tune thresholds
or train models.

## Capstone research status

The research workspace distinguishes implemented software from demonstrated
research findings. See
[`research/plan_to_implementation.md`](research/plan_to_implementation.md)
for the status of each proposal research question, experiment gate, dataset
integrity commitment, and advisor decision. No integrated-system accuracy or
ablation result is claimed until an independently labeled scenario set exists.
