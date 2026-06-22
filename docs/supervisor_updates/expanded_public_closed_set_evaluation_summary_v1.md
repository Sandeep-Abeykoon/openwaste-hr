# Expanded Public Closed-Set Evaluation Summary v1

## Purpose

This stage evaluates Baseline C with closed-set classification metrics.

## Baseline C

```text id="un7us3"
Baseline C = pretrained expanded public dataset model
```

## Evaluation Input

| Item          | Path                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| checkpoint    | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_best.pt            |
| class mapping | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_class_mapping.json |
| test manifest | ml/data/splits/expanded_public_known_test_v1.csv                                                      |

## Current Training Result

| Metric                   |  Value |
| ------------------------ | -----: |
| best epoch               |     18 |
| best validation macro-F1 | 0.8927 |
| test accuracy            | 0.8876 |
| test macro-F1            | 0.8819 |

## Key Research Point

The expanded public model has similar accuracy to the TrashNet-only pretrained model, but it improves macro-F1.

This matters because the expanded model is trained and evaluated on a broader six-class setting that includes organic waste.

## Next Stage

After this stage, the expanded public model should be tested with reject-option and unknown-handling evaluations.
