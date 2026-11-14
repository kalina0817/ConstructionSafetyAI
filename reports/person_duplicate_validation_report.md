# Person Detection Duplicate Filter Validation

**Validation date:** 2026-10-31  
**Project:** Context-Aware Vision Reasoning for Proactive Construction Workplace Safety

## 1. Purpose

This validation examines whether the person-detection duplicate filter reduces overlapping person bounding-box predictions in construction-site images.

The filter is intended to reduce possible duplicate detections. It does not identify people or establish that two detections represent the same individual.

## 2. Validation Method

- Images tested: 20
- Dataset: construction PPE test images
- Detector comparison: person detections from the same detector
- IoU suppression threshold: 0.60
- Containment suppression threshold: 0.85
- Higher-confidence detections are retained when a possible duplicate is identified.
- Original detections are preserved in the inference JSON reports.

A candidate is flagged when its overlap with a higher-confidence person box meets either threshold.

## 3. Results

| Metric | Result |
|---|---:|
| Images tested | 20 |
| Raw person detections | 35 |
| Filtered person detections | 32 |
| Possible duplicates suppressed | 3 |
| Images with suppressions | 1 |
| Images without suppressions | 19 |
| Total detections across all models | 207 |

The filter suppressed 3 of 35 raw person detections in this sample. The remaining 32 person detections were retained.

## 4. Per-Image Results

| Image | Raw people | Filtered people | Possible duplicates |
|---|---:|---:|---:|
| image110_jpg.rf.646d100f4c76a8d8691e383e76b63cff.jpg | 6 | 3 | 3 |
| image1139_jpg.rf.078c084034a008b231b7e4e2b47cb6b5.jpg | 1 | 1 | 0 |
| image1229_jpg.rf.e4b3045b748ad00d306e7e6dab0cf059.jpg | 1 | 1 | 0 |
| image1270_jpg.rf.ca9ddd1d29bd213b0a5706d94e9ce030.jpg | 2 | 2 | 0 |
| image1332_jpg.rf.2b417681a60d9a30dbe329a0e96c092d.jpg | 1 | 1 | 0 |
| image160_jpg.rf.eb8de852ca8ed59b047ed39ab8f1f8c6.jpg | 1 | 1 | 0 |
| image190_jpg.rf.9c6664e9eca489d09bbe67e44b16a7c0.jpg | 1 | 1 | 0 |
| image287_jpg.rf.3c575351365790ec9f629fb54cfab200.jpg | 1 | 1 | 0 |
| image389_jpg.rf.ad68644f892ae5318d77ee08cc0128b1.jpg | 1 | 1 | 0 |
| image444_jpg.rf.453690299ff3b075b431f67d515b3c7e.jpg | 1 | 1 | 0 |
| image484_jpg.rf.4fafb415503a874d1cbb2f607434666b.jpg | 1 | 1 | 0 |
| image538_jpg.rf.b121211228e2c6ce58d1e957ca169334.jpg | 5 | 5 | 0 |
| image611_jpg.rf.fb2b8dd3a2e0c0a11dda0b903e3aba2b.jpg | 5 | 5 | 0 |
| image696_jpg.rf.1210505beb7c0809f11b8d912bb38fa8.jpg | 2 | 2 | 0 |
| image739_jpeg.rf.e3aa00c72ccce364f411311406289200.jpg | 1 | 1 | 0 |
| image810_jpg.rf.5b86489c944dce6f70dada0e743370ec.jpg | 1 | 1 | 0 |
| image910_jpg.rf.e8fa5d75e66a21ab3e43f39e619e8286.jpg | 1 | 1 | 0 |
| image935_jpg.rf.4ee9e0aa08d35780e6ce9aa5576d0562.jpg | 1 | 1 | 0 |
| image1_jpeg.rf.9ca4ae5023090ea377ed260fe1d14423.jpg | 1 | 1 | 0 |
| image1023_jpg.rf.a47e9f23a9dc73c328dd88a9655d8343.jpg | 1 | 1 | 0 |

## 5. Manual Review

**Image reviewed:** image110

Visual inspection found that six raw person bounding boxes corresponded to three actual workers. The filter retained three person detections for this image.

This manual review supports the filter's behavior on this particular image. It does not establish performance on all images.

### 5.1 Suppression Details for image110

The filter suppressed three lower-confidence person detections. All three comparisons involved detections from the PPE model.

| Suppression | Candidate confidence | Kept confidence | IoU | Containment | Suppression reason |
|---|---:|---:|---:|---:|---|
| 1 | 0.47760 | 0.71933 | 0.5880 | 0.8680 | Containment |
| 2 | 0.39315 | 0.71933 | 0.3589 | 0.9722 | Containment |
| 3 | 0.35774 | 0.81272 | 0.6300 | 0.9875 | IoU |

The first two candidates met the containment threshold of 0.85. The third met the IoU threshold of 0.60. In all three cases, the filter retained the higher-confidence detection.

The visual review found that the six raw person boxes represented three actual workers. The three retained detections were consistent with that observation for this image.

## 6. Limitations

- This is a preliminary 20-image sample, not a full test-set evaluation.
- Images without suppressions were not individually confirmed to have perfectly accurate person counts.
- Possible duplicate suppression is based on bounding-box overlap, not verified person identity.
- Overlapping boxes may sometimes belong to different people; visual review remains necessary.
- Detector predictions do not prove PPE compliance or the absence of PPE.
- Image-space relationships do not establish real-world distance, contact, or a confirmed hazard.

## 7. Conclusion

In this 20-image sample, the duplicate filter reduced person detections from 35 to 32 by suppressing 3 possible duplicates. All 20 inference reports were present, and the consolidated CSV totals were independently verified.

The results provide preliminary evidence that the filter can reduce overlapping person predictions. Further validation on a larger sample, with manual review of both suppressed and retained detections, is needed before making broader accuracy claims.

## 8. Supporting Files

- `person_duplicate_validation_20_images.csv` : per-image summary
- `person_duplicate_validation_20_images.json` : detailed validation data
- `inference_runs/` : original inference reports and detection evidence

