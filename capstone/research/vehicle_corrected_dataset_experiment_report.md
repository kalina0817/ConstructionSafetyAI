# Corrected Vehicle Detection Experiment Report

**Project:** ConstructionSiteSafetyAI  
**Experiment:** YOLO11n — Corrected Vehicle Dataset  
**Date:** November 12, 2026

## 1. Objective

Evaluate a YOLO11n vehicle detector using a structurally audited and corrected vehicle dataset. The experiment addresses identified annotation errors and exact duplicate training images while preserving the original dataset.

This is a construction-site hazard-screening research system, not a safety certification system.

## 2. Dataset preparation

Corrected dataset:
`data/processed/construction-vehicle-detection-clean-loader-corrected`

The original clean dataset was preserved.

### Annotation corrections

Five images previously labeled as Roller were verified to contain Loaders. Their labels were corrected to class ID 6 (Loader) in the separate corrected dataset:

- Four training images
- One validation image
- No test images

The 13 other full-image Roller labels examined in this audit were retained as Roller.

### Duplicate cleanup

Three exact duplicate ForkLift image pairs were identified in the training split.

- One image from each pair was retained.
- The redundant image and label files were moved to quarantine.
- The retained annotations were updated using the average of the two normalized YOLO boxes.
- Original files were preserved in the quarantine folder.

Quarantine:
`capstone/research/duplicate_forklift_quarantine`

Cleanup record:
`capstone/research/duplicate_forklift_cleanup.json`

### Final dataset integrity

| Split | Images | Labels | Objects | Empty labels |
|---|---:|---:|---:|---:|
| Train | 12,454 | 12,454 | 18,699 | 6 |
| Validation | 2,670 | 2,670 | 3,974 | 0 |
| Test | 2,670 | 2,670 | 4,021 | 1 |

The structural audit reported:

- Malformed labels: 0
- Invalid class IDs: 0
- Invalid coordinates: 0
- Unreadable images: 0
- Missing labels: 0
- Missing images: 0
- Remaining exact duplicate image groups: 0

**Structural audit result: PASS**

The empty labels were retained. No unsupported changes were made to those annotations.

## 3. Training configuration

| Setting | Value |
|---|---|
| Model | YOLO11n |
| Initial pretrained weights | `yolo11n.pt` |
| Dataset YAML | `data/processed/construction-vehicle-detection-clean-loader-corrected/data.yaml` |
| Epochs | 5 |
| Image size | 640 |
| Batch size | 4 |
| Device | CPU |
| Workers | 0 |
| Optimizer | AdamW |
| Learning rate | 0.000714 |
| Weight decay | 0.0005 |
| Seed | 42 |
| Deterministic | True |
| Validation | Enabled |
| Validation split | val |
| IoU | 0.7 |
| Mixed precision | Disabled |

Training was interrupted and subsequently resumed from `last.pt`. The saved results CSV therefore contains duplicate records for epochs 4 and 5. These are retained as recorded and do not represent seven distinct epochs.

Training output:
`runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep`

Selected checkpoint for final evaluation:
`runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt`

## 4. Validation history

The saved CSV contains the following records. Duplicate epoch entries are preserved because they document the interrupted and resumed run.

| Epoch record | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| 1 | 82.34% | 63.76% | 71.17% | 53.74% |
| 2 | 87.70% | 67.78% | 78.64% | 58.82% |
| 3 | 87.41% | 75.97% | 83.88% | 63.75% |
| 4 (initial) | 91.67% | 79.79% | 88.26% | 70.24% |
| 4 (resumed) | 89.47% | 80.34% | 87.89% | 70.86% |
| 5 (initial) | 90.20% | 85.05% | 91.13% | 74.63% |
| 5 (resumed) | 91.42% | 85.38% | 91.20% | 74.92% |

Final recorded validation metrics: precision 91.42%, recall 85.38%, mAP@50 91.20%, and mAP@50–95 74.92%.

## 5. Final held-out test evaluation

The final evaluation used `best.pt` and the corrected dataset's test split.

| Evaluation setting | Value |
|---|---|
| Model | YOLO11n `best.pt` |
| Split | test |
| Test images | 2,670 |
| Image size | 640 |
| Batch size | 4 |
| Device | CPU |
| Workers | 0 |

### Overall test metrics

| Metric | Result |
|---|---:|
| Precision | 92.31% |
| Recall | 85.17% |
| mAP@50 | 91.74% |
| mAP@50–95 | 75.52% |

### Per-class test metrics

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Boringmachine | 92.2% | 85.3% | 92.0% | 75.1% |
| Bulldozer | 95.0% | 92.2% | 97.2% | 79.5% |
| Dump Truck | 86.4% | 74.4% | 86.4% | 64.4% |
| Excavator | 93.2% | 80.5% | 91.2% | 69.1% |
| ForkLift | 97.2% | 99.3% | 99.5% | 88.3% |
| Grader | 96.7% | 96.1% | 98.8% | 82.3% |
| Loader | 86.4% | 76.9% | 87.2% | 71.3% |
| Mixer Truck | 94.7% | 76.9% | 81.7% | 72.2% |
| Mobile Crane | 84.5% | 79.8% | 86.5% | 61.2% |
| Roller | 96.9% | 90.4% | 97.0% | 91.8% |

ForkLift achieved the highest mAP@50 at 99.5%. Roller achieved the highest mAP@50–95 at 91.8%. Mobile Crane had the lowest mAP@50–95 at 61.2%. Dump Truck had the lowest recall at 74.4%.

These class-level results can guide later error analysis. The held-out test set must not be used for model tuning.

## 6. Evaluation artifacts

Final test evaluation directory:

`runs/detect/runs/detect/vehicle_corrected_dataset_final_test_evaluation`

Saved artifacts include:

- `BoxF1_curve.png`
- `BoxPR_curve.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `val_batch0_labels.jpg` and `val_batch0_pred.jpg`
- `val_batch1_labels.jpg` and `val_batch1_pred.jpg`
- `val_batch2_labels.jpg` and `val_batch2_pred.jpg`

Training history and plots are stored in the training output directory, including `args.yaml`, `results.csv`, `results.png`, and the training and validation batch images.

## 7. Interpretation and limitations

The corrected-dataset experiment achieved 91.74% mAP@50 and 75.52% mAP@50–95 on the held-out test split.

Results should be interpreted in light of the dataset, annotation quality, and the five-epoch training schedule. Performance is not uniform across classes. Mobile Crane, Dump Truck, and Loader are candidates for further error analysis.

The evaluation does not establish real-world hazard detection accuracy, worker-equipment interaction understanding, or safety certification. Object detection alone does not prove that a detected vehicle or person is involved in a hazardous situation.

Any future model changes should be evaluated using a clearly documented experimental protocol. Do not tune against the held-out test results.

## 8. Reproducibility references

- Corrected dataset: `data/processed/construction-vehicle-detection-clean-loader-corrected`
- Training configuration: `runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/args.yaml`
- Training history: `runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/results.csv`
- Best checkpoint: `runs/detect/runs/detect/vehicle_yolo11n_corrected_dataset_5ep/weights/best.pt`
- Final test artifacts: `runs/detect/runs/detect/vehicle_corrected_dataset_final_test_evaluation`
- Duplicate cleanup report: `capstone/research/duplicate_forklift_cleanup.json`
