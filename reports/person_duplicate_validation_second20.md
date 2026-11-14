# Person Duplicate Detection Validation - Second Batch

Validation date: October 31, 2026
Images processed: 20
IoU threshold: 0.60
Containment threshold: 0.85
Minimum area ratio: 0.35

## Results

| Measure | Result |
|---|---:|
| Images processed | 20 |
| Total detections across models | 224 |
| Raw person detections | 37 |
| Filtered person detections | 33 |
| Suppressed person detections | 4 |
| Images with suppressions | 2 |

## Per-image results

| Image | Run | Raw | Retained | Suppressed |
|---|---|---:|---:|---:|
| image1003_jpg.rf.f7682a7ed9f110bc3de1fbedf6dd2842.jpg | 20261031_164244 | 1 | 1 | 0 |
| image1065_jpg.rf.79fcff8959d9fdd25b047dd9c6b01cea.jpg | 20261031_165438 | 1 | 1 | 0 |
| image1114_jpg.rf.84cded290e10763ff1da630433b16b7c.jpg | 20261031_165642 | 1 | 1 | 0 |
| image1152_jpg.rf.b1942a172db696b3103d6914b0f722f7.jpg | 20261031_165827 | 1 | 1 | 0 |
| image1235_jpg.rf.f6202afedaf32b17113aa8ae5cee3a18.jpg | 20261031_170024 | 3 | 3 | 0 |
| image1293_jpg.rf.3a193224cf5869033f8e23cd43513a8c.jpg | 20261031_170230 | 1 | 1 | 0 |
| image1406_jpg.rf.9335b0d8e48a5ad489bad26122a554fc.jpg | 20261031_170500 | 1 | 1 | 0 |
| image168_jpg.rf.daa12885200b0fc211f7d40dc5c3d4ce.jpg | 20261031_170658 | 1 | 1 | 0 |
| image241_jpg.rf.29c1ae54f0076991b72e10ffdfc8f6fb.jpg | 20261031_170904 | 1 | 1 | 0 |
| image350_jpg.rf.71267bd9d859ff1d87e3f0e221c954e6.jpg | 20261031_171128 | 1 | 1 | 0 |
| image423_jpg.rf.fa959934bb912e82428ef8cd8616be18.jpg | 20261031_171311 | 1 | 1 | 0 |
| image462_jpg.rf.4c2d550e4847195f801bdc059996ebc0.jpg | 20261031_192403 | 7 | 5 | 2 |
| image536_jpg.rf.b747b640a97614cc00d508576c3d0371.jpg | 20261031_171727 | 6 | 6 | 0 |
| image58_jpeg.rf.bd3970755a6b6567327123c2292d36e3.jpg | 20261031_171935 | 1 | 1 | 0 |
| image694_jpeg.rf.182fcd86d17bb2548fca863570f50555.jpg | 20261031_172116 | 1 | 1 | 0 |
| image722_jpeg.rf.2d1351263bdb4e266ac029bb31c11fed.jpg | 20261031_172355 | 2 | 2 | 0 |
| image805_jpg.rf.ed7bc2760b2322d66b4602a49ff985de.jpg | 20261031_195720 | 4 | 2 | 2 |
| image905_jpg.rf.cc67acf27afb32dd1059a9cfbb37519b.jpg | 20261031_173015 | 1 | 1 | 0 |
| image943_jpg.rf.749855805d122121354988f70a47a066.jpg | 20261031_173307 | 1 | 1 | 0 |
| image999_jpg.rf.7d04fc38434a0538e80aaf52c52694d9.jpg | 20261031_173615 | 1 | 1 | 0 |

## Manual review

- image462: visually checked; five actual people were retained.
- image805: visually checked; two actual people were retained.

## Scope and limitations

This is a preliminary 20-image validation sample. Results assess duplicate-person filtering on these images and do not establish performance on the full dataset.
