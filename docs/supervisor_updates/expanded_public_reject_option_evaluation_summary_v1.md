# Expanded Public Reject-Option Evaluation Summary v1

## Purpose

This stage evaluates reject-option behaviour for the expanded public pretrained model.

## Baseline C

```text id="6l60gl"
Baseline C = pretrained expanded public dataset model
```

## Input Data

| Split      | File                                             |
| ---------- | ------------------------------------------------ |
| validation | ml/data/splits/expanded_public_known_val_v1.csv  |
| test       | ml/data/splits/expanded_public_known_test_v1.csv |

## Methods Evaluated

| Method               | Role                               |
| -------------------- | ---------------------------------- |
| confidence threshold | softmax-based selective prediction |
| max-logit score      | logit-based selective prediction   |
| energy score         | energy-based selective prediction  |

## Current Closed-Set Result

| Metric            |  Value |
| ----------------- | -----: |
| accuracy          | 0.8876 |
| balanced accuracy | 0.8750 |
| macro-F1          | 0.8819 |
| weighted-F1       | 0.8870 |

## Research Point

This evaluation checks whether the expanded public model can improve not only classification accuracy, but also safer prediction behaviour through reject-option decision-making.

## Next Stage

The next stage should evaluate local unknown and public unknown/future-class handling.
