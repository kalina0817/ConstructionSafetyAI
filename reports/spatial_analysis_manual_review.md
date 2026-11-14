# Spatial Analysis Manual Review

## Purpose

This report records manual visual reviews of object detections and spatial relationships. It is separate from the person-duplicate validation report.

## Manual Review Results

| Image | Run ID | Actual people | Confirmed person detections | Actual machinery | Machinery detections | Generated relationships | Review findings |
|---|---|---:|---:|---:|---:|---:|---|
| image536 | 20261031_202047 | 6 | 6 | 0 | 4 Mobile Crane | 24 | All four Mobile Crane detections were false positives. The 24 relationships are based on false-positive machinery detections. |
| image999 | 20261031_202142 | 1 | 1 | 0 | 1 Roller | 1 | Roller detection was a false positive. The relationship is based on the false-positive Roller detection. |
| image943 | 20261031_202133 | 1 | 1 | 0 | 0 | 0 | One worker confirmed; no vehicles or heavy machinery visible. No spatial relationships generated. |
| image905 | 20261031_202126 | 1 | 1 | 0 | 1 Roller | 1 | Roller detection was a false positive. The relationship is based on the false-positive Roller detection. |
| image722 | 20261031_202110 | 2 | 2 | 0 | 1 Roller | 2 | Roller detection was a false positive. Both generated relationships are based on the false-positive Roller detection. |
| image694 | 20261031_202102 | 1 | 1 | 0 | 2 Roller | 2 | Both Roller detections were false positives. Both generated relationships are based on false-positive machinery detections. |
| image423 | 20261031_202032 | 1 | 1 | 0 | 0 | 0 | One worker confirmed. No vehicles or heavy machinery visible. No spatial relationships generated. |
| image462 | 20261031_202039 | 5 | 5 | 0 | 0 | 0 | Five workers confirmed. No vehicles or heavy machinery visible. Two suppressed person detections were previously confirmed as duplicates. No spatial relationships generated. |
| image805 | 20261031_202118 | 2 | 2 | 0 | 0 | 0 | Two workers confirmed. No vehicles or heavy machinery visible. Two suppressed person detections were previously confirmed as duplicates. No spatial relationships generated. |
| image58 | 20261031_202055 | 1 | 1 | 0 | 0 | 0 | One worker confirmed. No vehicles or heavy machinery visible. No spatial relationships generated. |
| image110 | 20261031_201632 | 3 | 3 | 0 | 0 | 0 | Three actual workers confirmed. Three person detections retained; three suppressed detections were confirmed as duplicates. No machinery or spatial relationships detected. |

## Interpretation and Limitations

Spatial relationships are generated from detector predictions. A generated relationship does not establish actual physical distance, contact, unsafe proximity, or a confirmed hazard.

Manual visual review findings apply only to the images reviewed. They do not establish overall detector accuracy.









