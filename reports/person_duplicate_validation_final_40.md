# Person Duplicate Filtering Validation ? 40 Images

## Validation configuration

- IoU threshold: 0.60
- Containment threshold: 0.85
- Minimum bounding-box area ratio: 0.35
- Images: 40 unique test images
- Inference reports: newest report selected for each image

## Overall results

- Raw person detections: 72
- Retained person detections: 65
- Suppressed possible duplicates: 7
- Total detections across models: 431

## Per-image results

| Image | Raw persons | Retained | Suppressed | Total detections |
|---|---:|---:|---:|---:|
| image110_jpg.rf.646d100f4c76a8d8691e383e76b63cff.jpg | 6 | 3 | 3 | 16 |
| image1139_jpg.rf.078c084034a008b231b7e4e2b47cb6b5.jpg | 1 | 1 | 0 | 12 |
| image1229_jpg.rf.e4b3045b748ad00d306e7e6dab0cf059.jpg | 1 | 1 | 0 | 7 |
| image1270_jpg.rf.ca9ddd1d29bd213b0a5706d94e9ce030.jpg | 2 | 2 | 0 | 8 |
| image1332_jpg.rf.2b417681a60d9a30dbe329a0e96c092d.jpg | 1 | 1 | 0 | 11 |
| image160_jpg.rf.eb8de852ca8ed59b047ed39ab8f1f8c6.jpg | 1 | 1 | 0 | 10 |
| image190_jpg.rf.9c6664e9eca489d09bbe67e44b16a7c0.jpg | 1 | 1 | 0 | 6 |
| image287_jpg.rf.3c575351365790ec9f629fb54cfab200.jpg | 1 | 1 | 0 | 9 |
| image389_jpg.rf.ad68644f892ae5318d77ee08cc0128b1.jpg | 1 | 1 | 0 | 7 |
| image444_jpg.rf.453690299ff3b075b431f67d515b3c7e.jpg | 1 | 1 | 0 | 10 |
| image484_jpg.rf.4fafb415503a874d1cbb2f607434666b.jpg | 1 | 1 | 0 | 4 |
| image538_jpg.rf.b121211228e2c6ce58d1e957ca169334.jpg | 5 | 5 | 0 | 23 |
| image611_jpg.rf.fb2b8dd3a2e0c0a11dda0b903e3aba2b.jpg | 5 | 5 | 0 | 24 |
| image696_jpg.rf.1210505beb7c0809f11b8d912bb38fa8.jpg | 2 | 2 | 0 | 14 |
| image739_jpeg.rf.e3aa00c72ccce364f411311406289200.jpg | 1 | 1 | 0 | 8 |
| image810_jpg.rf.5b86489c944dce6f70dada0e743370ec.jpg | 1 | 1 | 0 | 7 |
| image910_jpg.rf.e8fa5d75e66a21ab3e43f39e619e8286.jpg | 1 | 1 | 0 | 9 |
| image935_jpg.rf.4ee9e0aa08d35780e6ce9aa5576d0562.jpg | 1 | 1 | 0 | 9 |
| image1_jpeg.rf.9ca4ae5023090ea377ed260fe1d14423.jpg | 1 | 1 | 0 | 5 |
| image1023_jpg.rf.a47e9f23a9dc73c328dd88a9655d8343.jpg | 1 | 1 | 0 | 8 |
| image1003_jpg.rf.f7682a7ed9f110bc3de1fbedf6dd2842.jpg | 1 | 1 | 0 | 9 |
| image1065_jpg.rf.79fcff8959d9fdd25b047dd9c6b01cea.jpg | 1 | 1 | 0 | 10 |
| image1114_jpg.rf.84cded290e10763ff1da630433b16b7c.jpg | 1 | 1 | 0 | 10 |
| image1152_jpg.rf.b1942a172db696b3103d6914b0f722f7.jpg | 1 | 1 | 0 | 8 |
| image1235_jpg.rf.f6202afedaf32b17113aa8ae5cee3a18.jpg | 3 | 3 | 0 | 30 |
| image1293_jpg.rf.3a193224cf5869033f8e23cd43513a8c.jpg | 1 | 1 | 0 | 8 |
| image1406_jpg.rf.9335b0d8e48a5ad489bad26122a554fc.jpg | 1 | 1 | 0 | 8 |
| image168_jpg.rf.daa12885200b0fc211f7d40dc5c3d4ce.jpg | 1 | 1 | 0 | 4 |
| image241_jpg.rf.29c1ae54f0076991b72e10ffdfc8f6fb.jpg | 1 | 1 | 0 | 9 |
| image350_jpg.rf.71267bd9d859ff1d87e3f0e221c954e6.jpg | 1 | 1 | 0 | 9 |
| image423_jpg.rf.fa959934bb912e82428ef8cd8616be18.jpg | 1 | 1 | 0 | 7 |
| image462_jpg.rf.4c2d550e4847195f801bdc059996ebc0.jpg | 7 | 5 | 2 | 21 |
| image536_jpg.rf.b747b640a97614cc00d508576c3d0371.jpg | 6 | 6 | 0 | 28 |
| image58_jpeg.rf.bd3970755a6b6567327123c2292d36e3.jpg | 1 | 1 | 0 | 6 |
| image694_jpeg.rf.182fcd86d17bb2548fca863570f50555.jpg | 1 | 1 | 0 | 10 |
| image722_jpeg.rf.2d1351263bdb4e266ac029bb31c11fed.jpg | 2 | 2 | 0 | 8 |
| image805_jpg.rf.ed7bc2760b2322d66b4602a49ff985de.jpg | 4 | 2 | 2 | 9 |
| image905_jpg.rf.cc67acf27afb32dd1059a9cfbb37519b.jpg | 1 | 1 | 0 | 10 |
| image943_jpg.rf.749855805d122121354988f70a47a066.jpg | 1 | 1 | 0 | 9 |
| image999_jpg.rf.7d04fc38434a0538e80aaf52c52694d9.jpg | 1 | 1 | 0 | 11 |

## Interpretation and limitations

This report summarizes detector outputs and duplicate-filtering results. A suppressed detection is a possible duplicate according to the configured bounding-box criteria; it is not, by itself, proof that the detection was incorrect. Automated counts do not replace visual review of annotated images.

The metadata correction records the 0.35 minimum area ratio used by the filtering function and inference call. Original JSON files were backed up before their threshold metadata was updated.
## Manual Visual Review

Five of the 40 images received manual visual review.

| Image | Actual people identified | Detections retained | Suppressed detections | Review result |
|---|---:|---:|---:|---|
| image110 | 3 | 3 | 3 | Tentative: the retained detections appeared to correspond to the three workers identified |
| image1139 | 1 | 1 | 0 | Confirmed: one actual person was confirmed; no duplicate detections were observed |
| image462 | 5 | 5 | 2 | Confirmed: both suppressed detections appeared to be duplicates |
| image805 | 2 | 2 | 2 | Confirmed: both suppressed detections appeared to be duplicates |
| image536 | 6 | 6 | 0 | Confirmed: all six people were correctly detected; four Mobile Crane detections were false positives because no construction vehicles were visible |

The review covered all seven suppressed person detections in image110, image462, and image805. The image110 assessment remains tentative; image462 and image805 were confirmed.

Image536 was additionally reviewed for person and machinery detections: all six people were confirmed, while all four Mobile Crane detections were false positives. The 24 spatial relationships involving those false-positive machinery detections should not be interpreted as actual worker–machinery relationships.

The remaining 35 images were not manually reviewed.

Automated duplicate suppression and manual visual review are separate checks. These results do not establish that every person detection across all 40 images is correct.
