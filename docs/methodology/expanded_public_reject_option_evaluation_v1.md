# Expanded Public Reject-Option Evaluation v1

## Purpose

This stage evaluates reject-option behaviour for Baseline C.

Baseline C is the pretrained expanded public dataset model trained using combined TrashNet-style and RealWaste known samples.

## Baseline C Definition

```text id="y3i2ot"
Baseline C = pretrained expanded public dataset model
```

## Evaluation Goal

The goal is to test whether the expanded public pretrained model can make safer selective predictions by accepting high-confidence known samples and rejecting lower-confidence samples.

## Input Model

| Item          | Path                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| checkpoint    | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_best.pt            |
| class mapping | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_class_mapping.json |

## Input Manifests

| Split      | File                                             |
| ---------- | ------------------------------------------------ |
| validation | ml/data/splits/expanded_public_known_val_v1.csv  |
| test       | ml/data/splits/expanded_public_known_test_v1.csv |

## Evaluation Methods

This stage evaluates three reject-option methods:

| Method               | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| confidence threshold | accepts predictions when maximum softmax confidence is high enough |
| max-logit score      | accepts predictions when maximum logit score is high enough        |
| energy score         | accepts predictions based on energy-score thresholding             |

## Metrics

The evaluation reports:

* selected threshold
* coverage
* rejection rate
* selective accuracy
* selective error
* selective macro-F1
* selective weighted-F1

## Research Meaning

The closed-set expanded public model achieved:

| Metric            |  Value |
| ----------------- | -----: |
| accuracy          | 0.8876 |
| balanced accuracy | 0.8750 |
| macro-F1          | 0.8819 |
| weighted-F1       | 0.8870 |

However, OpenWaste-HR is not only about closed-set accuracy. The project also needs to evaluate whether the model can avoid unsafe predictions by rejecting uncertain cases.

## Next Stage

After this reject-option evaluation, the next stage should evaluate the expanded public pretrained model against local unknown and public unknown/future-class samples.
