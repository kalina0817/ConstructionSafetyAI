# Context-aware vision reasoning for construction safety

Independent master's capstone workspace for evaluating whether explicit,
image-grounded contextual reasoning improves construction-site image
screening over object detection alone.

This project is research decision support, not a safety-certification,
regulatory-compliance, or autonomous control system. Detector predictions and
image-space geometry do not prove PPE absence, machine operation, physical
distance, an unsafe condition, or regulatory noncompliance.

## Clean restart boundary

This folder is the clean capstone project. It references the existing local
datasets, selected model checkpoints, and training records in the parent
workspace without copying or modifying them. Existing files outside this
folder are deliberately left untouched until this replacement project is
reviewed and validated.

Datasets and checkpoints remain local artifacts. The PPE dataset metadata
identifies its source license as private. Confirm dataset and checkpoint
redistribution rights with the advisor before publishing any data, trained
weights, or derived material.

## Start the application

From the project directory:

```powershell
cd capstone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The app includes a research overview, single-image screening workflow, and
independently labeled scenario evaluation workspace. Model artifacts are
loaded from the configured parent workspace; no weights are downloaded.
PPE/toolset and vehicle detectors have separate confidence thresholds. The
vehicle threshold starts at 0.40 because a manually reviewed image produced
four false Mobile Crane detections at scores 0.25–0.37, which were absent at
0.40. This is a provisional user-facing filter, not a calibrated or
research-validated operating point; assess thresholds on a separate validation
set before making performance claims.

## Reproduce detector validation

Run validation data during development:

```powershell
python scripts\evaluate_detector.py --model ppe --split val
python scripts\evaluate_detector.py --model toolset --split val
python scripts\evaluate_detector.py --model vehicles --split val
```

The test split is locked by default. After model selection, confidence
settings, and all other analysis choices are frozen, a single preregistered
test evaluation requires:

```powershell
python scripts\evaluate_detector.py --model ppe --split test --allow-locked-test
```

Every evaluation writes a new JSON record containing the dataset, checkpoint,
training-argument hashes, software environment, configuration, aggregate
metrics, and per-class metrics to `reports/detector_evaluations/`. Test results
must not be used to select models or tune thresholds.

## Audit the selected datasets

```powershell
python scripts\audit_datasets.py
```

The read-only audit checks all selected dataset splits, image readability,
image/label pairing, YOLO class IDs, normalized coordinates, and per-class
annotation counts. It reports images without label files as an ambiguity to
resolve; it does not silently treat them as either negative examples or
errors. Reports are written under `reports/dataset_audits/`.

Optional fine-tuning uses only the configured training split and validation
split:

```powershell
python scripts\train_detector.py --model ppe --initial-weights runs\detect\runs\detect\ppe_yolo11n_postcorrection_15ep\weights\best.pt --epochs 15 --seed 42
```

Training writes to a new capstone report directory and records initial-weight
and dataset hashes. It never writes into the preserved source checkpoint
directory and does not use the test split. Run only after the advisor-approved
data and training protocol is finalized.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

Tests do not load large model checkpoints or modify source datasets.

## Evaluation protocol

The integrated contextual-reasoning system requires an independent,
scenario-based annotation set. Its labels must not be derived from model
predictions. Keep training, validation, and test roles separate; group
near-duplicates and related image sequences in one partition. Record
provenance, permitted use, inclusion criteria, and annotation-guide version.

The evaluator supports image-level categories:

- `potential_missing_ppe`
- `potential_worker_machinery_interaction`

Each image/category pair requires a positive, negative, or explicit uncertain
label. Uncertain cases are excluded from binary metrics and reported. Model
absence alone is not a negative PPE label. Current evaluation does not measure
box localization or relationship endpoint correctness.

The current 0.05 normalized machinery edge-distance and 0.80 PPE containment
settings are exploratory geometry parameters. Tune only against a separate
validation set; never present them as calibrated risk or physical distance.

See [the research plan and status register](research/plan_and_status.md),
[preliminary literature review](research/literature_review.md),
[annotation guide](research/annotation_guide.md),
[evaluation protocol](research/evaluation_protocol.md), and
[thesis structure](research/thesis_outline.md).
