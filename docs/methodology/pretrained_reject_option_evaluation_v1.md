# Pretrained Reject-Option Evaluation v1

## Purpose

This stage evaluates reject-option behaviour using the full pretrained transfer-learning checkpoint.

The scratch-trained model is treated as:

```text id="dv25cp"
Baseline A = scratch-trained TrashNet-style model
```

The pretrained model is treated as:

```text id="jvlmbx"
Baseline B = pretrained transfer-learning model
```

The aim is to test whether the stronger pretrained model also improves uncertainty-aware selective prediction.

## Checkpoint Used

The pretrained checkpoint used in this stage is:

```text id="19mjx4"
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

## Config Files

This stage uses two pretrained reject-option config files:

| Config                                                | Purpose                                      |
| ----------------------------------------------------- | -------------------------------------------- |
| ml/configs/confidence_reject_pretrained_trashnet.yaml | confidence-threshold reject evaluation       |
| ml/configs/open_set_score_pretrained_trashnet.yaml    | max-logit and energy-score reject evaluation |

## Methods Evaluated

The reject-option methods are:

| Method               | Score Used                 | Decision Rule                                    |
| -------------------- | -------------------------- | ------------------------------------------------ |
| confidence threshold | maximum softmax confidence | accept if confidence is high enough              |
| max-logit score      | highest raw logit          | accept if max logit is high enough               |
| energy score         | energy value from logits   | accept if energy indicates sufficient confidence |

## Metrics

The main known-test metrics are:

| Metric             | Meaning                                                  |
| ------------------ | -------------------------------------------------------- |
| coverage           | proportion of known-test samples automatically accepted  |
| rejection rate     | proportion of known-test samples routed to manual review |
| selective accuracy | accuracy among accepted samples only                     |
| selective macro-F1 | macro-F1 among accepted samples only                     |

These metrics are important because OpenWaste-HR should not only improve closed-set accuracy. It should also improve the reliability of accepted decisions.

## Comparison with Scratch-Trained Baseline

The scratch-trained confidence reject baseline previously achieved:

| Metric                    |    Value |
| ------------------------- | -------: |
| Known-test coverage       | 0.682292 |
| Known-test rejection rate | 0.317708 |
| Selective accuracy        | 0.770992 |
| Selective macro-F1        | 0.715164 |

The pretrained model has stronger closed-set performance, so this stage checks whether its reject-option behaviour also improves.

## Research Meaning

This stage evaluates whether pretrained visual features improve selective prediction reliability.

If the pretrained reject-option result improves coverage and accepted-decision accuracy, it strengthens the case for using the pretrained model as the base classifier inside the final OpenWaste-HR hierarchical decision workflow.
