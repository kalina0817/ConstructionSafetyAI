# Construction Safety AI — 12-photo study

All 36 checks finished. The app used YOLO11n to find objects in all 12 photos. Each reasoning method received the same object boxes for a given photo. The study compares how well the methods described each worker near a particular machine.

**The main miss was a worker and an excavator bucket in the same narrow trench.** None of the three methods raised a linked alert for that pair. The detector missed the machine, so the app had no machine to connect to the worker. The photo supports a possible hazard. It does not show whether the bucket was moving.

## What each method did

| Method | Reviewed pairs it could check | Correct descriptions of worker position¹ | Machine alerts on reviewed pairs | Machine alerts across all photos² |
|---|---:|---:|---:|---:|
| Scene-graph rules | 6 of 16 | Checks proximity only | 6 | 15 |
| Qwen3-VL | 6 of 16 | 0 of 5 | 0 | 0 |
| MiniCPM-V | 6 of 16 | 1 of 5 | 0 | 2 |

¹ Five of the six linked pairs had a clear reference position. In the sixth, the loader hid the worker’s feet. A method that said it could not tell received no credit for an exact position match on the five clear pairs.

² The raw total includes boxes or pairs outside the review. Those alerts were not graded. All counts in this table concern workers and machines. Protective gear was checked separately.

**Scene-graph rules.** The rules flagged all six linked pairs. Three alerts were on pairs without visible evidence of a specific hazard. The other three were on pairs where the photo left the hazard unclear.

**Qwen3-VL.** The worker close-up check did not identify a person for any of the six reviewed pairs it could reach. The app therefore returned ‘cannot tell’ on all six. A completed run could still leave every one of these questions unanswered.

**MiniCPM-V.** It matched the reference position on 1 of five clear linked pairs. On photo 1, it placed the worker on the excavator but described standing on a step. The worker was seated in the cab. On photo 3, it placed a worker on the machine even though their feet were on rubble. Its two raw machinery alerts on photo 1 came from a different worker box than the reviewed cab occupant. Neither was scored as a confirmed hazard.

**How much the app could check.** Only six of 16 reviewed worker–machine pairs reached all three methods. The app missed the worker in six pairs and the machine in three. In one more pair it found both but did not connect them. It also gave the machine the wrong name in two of the six linked pairs. A missing pair never counts as a correct answer.

**What this establishes.** These are exploratory results from 12 selected photos with AI-assisted reference notes. Source captions were visible when those notes were made. There was one reference-supported hazard, four pairs without visible evidence of a specific hazard, and eleven where the hazard was unclear. Independent human review and a larger set of hazards are still needed to measure accuracy.

## Photo-by-photo results

Each row is one worker paired with one machine. ‘No alert’ describes the app’s action and can include uncertainty. It does not establish that a worker was safe.

| Photo | Worker | Reference position | Reference hazard | Rules | Qwen3-VL | MiniCPM-V |
|---|---|---|---|---|---|---|
| [01](../../reviewer/photos/Photo_01.jpg) | Dark jacket inside open cab | on the machine | no specific hazard visible | Alert | cannot tell · no alert | on the machine · no alert |
| [02](../../reviewer/photos/Photo_02.jpg) | Red shirt at rear left of cab | cannot tell | cannot tell | Alert | cannot tell · no alert | on the machine · no alert |
| [03](../../reviewer/photos/Photo_03.jpg) | Yellow helmet and dark clothing, left foreground | beside the machine | cannot tell | Alert | cannot tell · no alert | on the machine · no alert |
| [04](../../reviewer/photos/Photo_04.jpg) | Striped shirt behind cab glass | on the machine | no specific hazard visible | No linked pair | No linked pair | No linked pair |
| [04](../../reviewer/photos/Photo_04.jpg) | White shirt cropped at right edge | in the same scene | cannot tell | No linked pair | No linked pair | No linked pair |
| [05](../../reviewer/photos/Photo_05.jpg) | Purple shirt in open cab | on the machine | no specific hazard visible | Alert | cannot tell · no alert | cannot tell · no alert |
| [06](../../reviewer/photos/Photo_06.jpg) | Distant blue figure behind tarps, left | in the same scene | cannot tell | No linked pair | No linked pair | No linked pair |
| [07](../../reviewer/photos/Photo_07.jpg) | Red jacket just behind boom | in the same scene | cannot tell | No linked pair | No linked pair | No linked pair |
| [08](../../reviewer/photos/Photo_08.jpg) | Dark-clothed person at far right | beside the machine | no specific hazard visible | Alert | cannot tell · no alert | cannot tell · no alert |
| [09](../../reviewer/photos/Photo_09.jpg) | Small dark figure at far end of access road | in the same scene | cannot tell | No linked pair | No linked pair | No linked pair |
| [10](../../reviewer/photos/Photo_10.jpg) | Dark-clothed foreground worker, left of truck | beside the machine | cannot tell | Alert | cannot tell · no alert | on the machine · no alert |
| [11](../../reviewer/photos/Photo_11.jpg) | Blue-clothed person at pipe mouth in trench | in the same scene | cannot tell | No linked pair | No linked pair | No linked pair |
| [11](../../reviewer/photos/Photo_11.jpg) | Upright person on left lip of trench, behind pipe | in the same scene | cannot tell | No linked pair | No linked pair | No linked pair |
| [12](../../reviewer/photos/Photo_12.jpg) | Yellow-helmeted person inside narrow trench | in the same scene | possible hazard visible | No linked pair | No linked pair | No linked pair |
| [12](../../reviewer/photos/Photo_12.jpg) | Person in green trousers on upper bank | beside the machine | cannot tell | No linked pair | No linked pair | No linked pair |
| [12](../../reviewer/photos/Photo_12.jpg) | Person in light trousers on upper bank | beside the machine | cannot tell | No linked pair | No linked pair | No linked pair |

## Did the photo support the explanation?

Qwen3-VL gave no worker position to check on the six linked pairs. The rules used box positions and made no claim about the surface under the worker’s feet. The MiniCPM-V explanations are reviewed below.

| Photo | MiniCPM-V explanation | Evidence review |
|---|---|---|
| 01 | operator's feet on the machine step or deck | Contradicted for posture. The worker is seated in the open cab with bent knees. The close-up summary claimed a standing worker on a step. Being on the machine was correct. |
| 02 | feet on machine step or deck, person seated on equipment | Indeterminate. The loader hides the worker’s legs. The photo cannot establish a visible step under the feet or the claimed seat. |
| 03 | worker's feet are visible resting on the machine step | Contradicted. Both feet are on rubble beside the excavator. The claimed machine step does not support the worker. |
| 05 | The close-up does not establish an identifiable worker. | Abstained. The model declined to identify the worker. A person is visible in the blue excavator’s seat. |
| 08 | The close-up does not establish an identifiable worker. | Abstained. The model declined to identify the worker. The dark-clothed person at the right stands on dirt beside the bulldozer. |
| 10 | feet on machine step or deck, person seated on equipment | Contradicted. The worker is upright beside the roller’s rear wheel, with feet at ground level. No seat or step supports this person. |

## Protective gear and time taken

Each method returned five gear checks for each worker box, including ‘cannot tell’ when evidence was missing. Repeated boxes can describe the same person. The rules counts were calculated from the saved object records with the app’s gear checks. There were no independent gear labels to grade these answers.

| Method | Visible | May be missing | Cannot tell | Time for 12 photos³ |
|---|---:|---:|---:|---:|
| Scene-graph rules | 6 | 4 | 60 | Not measured separately |
| Qwen3-VL | 0 | 0 | 70 | 72.0 minutes |
| MiniCPM-V | 4 | 1 | 65 | 53.0 minutes |

³ Sum of the successful reasoning calls on this computer. Object detection was already complete. Photos with no worker boxes offered fewer checks.

## The trench photo

On photo 12, all three worker boxes were present. The detector missed the machine and its bucket. Qwen3-VL left the worker’s position unclear. MiniCPM-V mentioned a vehicle in its separate photo notes, but it described the person in the trench as standing on a machine step. The reference photo shows the worker in the trench. Neither visual method described the reviewed bucket exposure correctly.

## What to improve next

1. Use the trench photo to test whether the app can find a bucket or a partly visible machine and connect it to the correct worker.
2. Check worker close-ups that show a visible person but trigger ‘cannot tell’. Keep uncertain cases explicit.
3. Test seated workers and people standing on nearby ground. Require visible evidence for the claimed seat, step, or ground before describing their position.
4. Validate any changes on fresh, independently reviewed photos. These 12 photos have now been inspected.

## Files behind the findings

The study folder contains the two visual methods’ saved answers. The machine-readable comparison includes all 48 method and reference-pair rows, the evidence review, timings, and file fingerprints. The same three YOLO11n checkpoints supplied protective gear, tool, and vehicle detections in every run. Model IDs were qwen3-vl:4b-instruct and minicpm-v:latest.
