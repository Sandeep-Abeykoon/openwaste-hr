# Expanded Public Closed-Set Evaluation v1

## Purpose

This stage evaluates Baseline C using closed-set classification metrics.

Baseline C is the pretrained expanded public dataset model trained using the combined TrashNet-style and RealWaste known samples.

## Baseline C Definition

```text id="7p4zqk"
Baseline C = pretrained expanded public dataset model
```

## Evaluation Input

| Item             | Path                                                                                                  |
| ---------------- | ----------------------------------------------------------------------------------------------------- |
| model checkpoint | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_best.pt            |
| class mapping    | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_class_mapping.json |
| test manifest    | ml/data/splits/expanded_public_known_test_v1.csv                                                      |

## Evaluation Data

The expanded public known test split contains:

| Split               | Count |
| ------------------- | ----: |
| expanded known test |  1050 |

## Known Evaluation Classes

The closed-set evaluation uses six known classes:

| Fine Label      |
| --------------- |
| paper_cardboard |
| plastic         |
| glass           |
| metal           |
| organic         |
| residual        |

The `unknown` label is not part of closed-set evaluation.

## Training-Time Result to Confirm

The full training run reported:

| Metric                   |  Value |
| ------------------------ | -----: |
| best epoch               |     18 |
| best validation macro-F1 | 0.8927 |
| test accuracy            | 0.8876 |
| test macro-F1            | 0.8819 |

## Metrics to Report

This evaluation should generate:

* accuracy
* balanced accuracy
* macro-F1
* weighted-F1
* confusion matrix
* classification report
* test predictions

## Comparison Meaning

This stage prepares the fair comparison between:

| Model                            | Training Data                                       |
| -------------------------------- | --------------------------------------------------- |
| Scratch TrashNet baseline        | TrashNet-style dataset only                         |
| Pretrained TrashNet baseline     | TrashNet-style dataset only                         |
| Expanded public pretrained model | TrashNet-style dataset plus RealWaste known samples |

## Research Meaning

The key question is whether dataset expansion improves the model beyond the TrashNet-only pretrained baseline.

The earlier TrashNet-only pretrained model achieved approximately:

| Metric   |  Value |
| -------- | -----: |
| accuracy | 0.8880 |
| macro-F1 | 0.8510 |

The expanded public pretrained model achieved approximately:

| Metric   |  Value |
| -------- | -----: |
| accuracy | 0.8876 |
| macro-F1 | 0.8819 |

Although accuracy is almost unchanged, macro-F1 improved, which is important because the expanded dataset contains six known classes and includes organic waste.

## Next Stage

After this closed-set evaluation, the next stage should evaluate reject-option behaviour and local/public unknown handling for the expanded public pretrained model.
