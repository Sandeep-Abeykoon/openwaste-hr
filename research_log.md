# OpenWaste-HR Research Log

## Project Direction

Selected direction: OpenWaste-HR, a hierarchical open-set waste classification system with reject option, coarse-to-fine fallback, and local active learning.

## Why This Direction Was Selected

The project moves beyond standard closed-set waste classification. Instead of forcing every input into a known class, the system will classify known items, fall back to coarse categories when uncertain, and reject unknown or ambiguous waste items for manual review.

## Initial Research Gap

Most existing waste classification systems are evaluated as closed-set classifiers, while real waste streams are open-world and may contain unknown, mixed, damaged, contaminated, or locally unusual items.

## Initial Implementation Plan

1. Freeze project taxonomy.
2. Prepare dataset mapping files.
3. Build closed-set baseline model.
4. Add confidence/reject-option baseline.
5. Build hierarchical coarse/fine model.
6. Evaluate unknown detection.
7. Add local correction UI.
8. Run active-learning experiment.
9. Export model for deployment.
10. Build final demo and thesis-ready results.

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-06-18 | Selected OpenWaste-HR direction | Stronger novelty than standard classifier + XAI + advice system |
| 2026-06-18 | Repository structure created | Keeps ML, backend, frontend, data, documentation, and experiments organized |
| 2026-06-18 | Frozen Taxonomy v1 | Created 7 fine known classes, 4 coarse classes, and reserved unknown/manual-review labels |
| 2026-06-18 | Added dataset intake plan | Created dataset source plan, manifest template, label mapping template, and validation tests |

## Taxonomy v1 Summary

Known fine labels:

1. paper_cardboard
2. plastic
3. glass
4. metal
5. organic
6. e_waste_battery
7. residual

Known coarse labels:

1. recyclable
2. organic
3. hazardous
4. residual

Reserved labels:

- unknown
- manual_review

The unknown label is not used as a normal training class. It is used for open-set evaluation, rejection, manual review, and active learning.

## Dataset Intake v1 Summary

The project will not mix image datasets directly.

Each image must be tracked using a manifest row containing:

- source dataset
- original label
- OpenWaste-HR fine label
- OpenWaste-HR coarse label
- known or unknown status
- experiment usage role
- image path
- license or citation note

Dataset intake starts with a simple baseline source, then moves to harder open-world and local unknown evaluation.

Unknown samples are not used as normal known-class training data in the first baseline.