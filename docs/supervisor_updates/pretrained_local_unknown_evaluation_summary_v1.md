# Pretrained Local Unknown Evaluation Summary v1

## Purpose

This stage evaluates the pretrained transfer-learning model on the local unknown dataset.

## Why This Matters

The pretrained model is much stronger on known-test classification, but the project also needs to know whether it handles local unknown images safely.

OpenWaste-HR should not only maximise known-test accuracy. It should also reduce unsafe unknown acceptance.

## Model Context

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |
| Baseline B | pretrained transfer-learning model   |

## Config

```text
ml/configs/pretrained_local_unknown_evaluation.yaml
```

## Main Metrics

| Metric                        | Meaning                                                 |
| ----------------------------- | ------------------------------------------------------- |
| unknown_rejection_rate        | local unknown samples routed to rejection/manual review |
| unknown_false_acceptance_rate | local unknown samples accepted as known                 |
| accepted label distribution   | known labels assigned to accepted unknown images        |

## Scratch-Trained Reference Result

| Method               | Unknown Rejection Rate | Unknown False Acceptance Rate |
| -------------------- | ---------------------: | ----------------------------: |
| Confidence threshold |               0.350000 |                      0.650000 |
| Max-logit score      |               0.275000 |                      0.725000 |
| Energy score         |               0.200000 |                      0.800000 |

## Next Stage

After this evaluation, the next step is to apply hierarchical decision evaluation and safe policy tuning to the pretrained checkpoint.
