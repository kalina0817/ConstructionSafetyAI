# Preliminary literature review

## Review status and boundary

This is a targeted, preliminary review to guide the capstone design; it is not
a systematic review and does not establish a definitive research gap. It
focuses on construction-site PPE detection, worker interaction recognition,
and visual-relation-based hazard reasoning. A broader search is still needed
for uncertainty calibration, explanation quality, human factors, dataset
governance, and closely related public systems. The advisor should approve the
final scope and contribution before either is presented as novel.

## Annotated related-work matrix

| Study | Approach and reported scope | Relevance to this project | Limits on comparison |
|---|---|---|---|
| Delhi, Sankarlal, and Thomas (2020) | Transfer-learned YOLOv3 for hardhat and safety-jacket compliance, using 2,509 construction-site images; the paper reports 0.96 F1 on its test set. | Establishes an object-detection/PPE-monitoring baseline and demonstrates a workflow-oriented application. | Different data, labels, split, and implementation. Its reported score is not directly comparable with this project's detector metrics or integrated hazard labels. |
| Li, Wei, Han, Jiang, Wang, and Huang (2022) | Visual relationship detection and construction-safety ontology for hazard identification; represents object interactions as triples and connects visual facts with safety knowledge. | Directly shows that visual relationships and knowledge-based hazard reasoning have prior construction-safety research. | The project's current image-space distance and containment rules are a much narrower prototype; this work does not validate those rules or their current thresholds. |
| Li, Wu, Xue, and Guo (2023) | Combines YOLO object/sign detection and ST-GCN behavior recognition to identify unsafe worker interactions with machinery or materials. The paper reports improved accuracy over ST-GCN-only results for its three studied behavior types. | Shows the value of interaction context and the distinction between recognizing motion and interpreting worker-object interactions. | It uses temporal behavior recognition and task-specific labels; it is not an evaluation of this project's single-image relationships or supported categories. |
| Azizi, Koskinopoulou, and Petillot (2024) | Compares Faster R-CNN and few-shot object detection for construction-site safety-equipment detection. | Motivates comparison of perception methods and attention to data-labeling effort, not only detector accuracy. | Different architectures, datasets, and task setup; reported detector results cannot establish the value of contextual screening here. |

## Synthesis and project positioning

The literature already includes PPE object detection, worker-object
interaction recognition, and visual-relation/ontology-based hazard
identification. Therefore, neither object detection, scene graphs, nor
rule-based contextual reasoning alone should be presented as this capstone's
novel contribution.

The current implementation is narrower than several related systems: it
screens single images using existing PPE, tool, and vehicle detectors; derives
image-space relationships; applies a small set of explicit rules; preserves
evidence links and uncertainty; and requires human verification. Its
exploratory geometry thresholds are not physical distances or calibrated risk
scores. The code and unit tests establish implementation behavior, not
empirical safety performance.

A defensible **candidate evaluation question**, pending a comprehensive
related-work review and advisor approval, is whether a frozen, evidence-linked
single-image reasoning pipeline changes image/category screening performance
relative to simpler detection-only and direct-rule baselines on an
independently annotated scenario set. A paired evaluation could also examine
whether reviewers find its evidence and uncertainty more useful, but that
requires a separate blinded rubric study. These are questions to test, not
claims that the system improves accuracy, safety, or decision quality.

For a valid comparison, all conditions must use the same frozen detector
outputs, examples, annotation guide, and operating settings. The project
needs independent scenario labels before it can answer the integrated-system
question. Detector validation metrics alone cannot answer it. Any relationship
extraction claim also requires reference relationships and a pre-specified
endpoint-matching protocol.

## References

1. Delhi, V. S. K., Sankarlal, R., & Thomas, A. (2020). Detection of Personal
   Protective Equipment (PPE) Compliance on Construction Site Using Computer
   Vision Based Deep Learning Techniques. *Frontiers in Built Environment,
   6*, 136. <https://doi.org/10.3389/fbuil.2020.00136>
2. Li, Y., Wei, H., Han, Z., Jiang, N., Wang, W., & Huang, J. (2022). Computer
   Vision-Based Hazard Identification of Construction Site Using Visual
   Relationship Detection and Ontology. *Buildings, 12*(6), 857.
   <https://doi.org/10.3390/buildings12060857>
3. Li, P., Wu, F., Xue, S., & Guo, L. (2023). Study on the Interaction
   Behaviors Identification of Construction Workers Based on ST-GCN and YOLO.
   *Sensors, 23*(14), 6318. <https://doi.org/10.3390/s23146318>
4. Azizi, R., Koskinopoulou, M., & Petillot, Y. (2024). Comparison of Machine
   Learning Approaches for Robust and Timely Detection of PPE in Construction
   Sites. *Robotics, 13*(2), 31.
   <https://doi.org/10.3390/robotics13020031>

## Review tasks before finalizing Chapter 2

- Expand the search to uncertainty calibration, explainable AI, human review,
  safety decision support, and relevant public benchmarks.
- Record search databases, query strings, dates, eligibility criteria, and
  exclusions if the review is presented as systematic or scoping research.
- Read and verify each included full text; check primary data, evaluation
  protocols, limitations, and dataset licenses rather than comparing headline
  metrics across papers.
- Compare the proposal's stated contribution against the expanded review, then
  obtain advisor approval for final research questions and novelty wording.
