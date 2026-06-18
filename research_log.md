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
| 2026-06-18 | Added TrashNet manifest builder | Created first dataset intake script for folder-based closed-set baseline data |
| 2026-06-18 | Added data inspection pipeline | Created manifest-based image loader, image validation, label distribution summaries, and inspection figures |
| 2026-06-18 | Added baseline training pipeline | Created PyTorch dataset, label encoder, MobileNetV3 baseline model, and closed-set training script |

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

## TrashNet Intake v1 Summary

TrashNet is used as the first simple closed-set baseline dataset.

TrashNet original labels are mapped into OpenWaste-HR labels as follows:

| TrashNet Label | Fine Label | Coarse Label |
|---|---|---|
| cardboard | paper_cardboard | recyclable |
| paper | paper_cardboard | recyclable |
| plastic | plastic | recyclable |
| glass | glass | recyclable |
| metal | metal | recyclable |
| trash | residual | residual |

The generated manifest files are:

- ml/data/splits/trashnet_manifest_v1.csv
- ml/data/splits/known_train.csv
- ml/data/splits/known_val.csv
- ml/data/splits/known_test.csv

These files prepare the first baseline training stage.

## TrashNet Data Inspection v1 Summary

The first real TrashNet manifest contains:

| Usage | Count |
|---|---:|
| known_train | 1766 |
| known_val | 377 |
| known_test | 384 |

Fine-label distribution:

| Fine Label | Count |
|---|---:|
| paper_cardboard | 997 |
| glass | 501 |
| plastic | 482 |
| metal | 410 |
| residual | 137 |

Research observation:

TrashNet is suitable for the first closed-set baseline because it provides simple folder-based waste images. However, it has important limitations for OpenWaste-HR:

1. It does not contain organic waste as a dedicated class.
2. It does not contain e-waste or battery waste as a dedicated class.
3. The residual/trash class is much smaller than the other main material classes.
4. It does not directly test unknown rejection or local adaptation.

Therefore, TrashNet will be used only for the first baseline pipeline. Later stages must add additional dataset sources and local unknown images.

## Baseline Training v1 Summary

The first training pipeline prepares a closed-set baseline classifier using TrashNet.

This baseline predicts only the fine labels currently available from TrashNet:

1. paper_cardboard
2. plastic
3. glass
4. metal
5. residual

The baseline does not yet include:

1. organic
2. e_waste_battery
3. unknown rejection
4. coarse fallback
5. active learning

This is intentional. The closed-set baseline is needed so that the later OpenWaste-HR method can be compared against a normal forced-classification model.