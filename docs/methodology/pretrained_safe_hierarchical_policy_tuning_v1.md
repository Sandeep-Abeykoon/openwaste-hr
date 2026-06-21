# Safe Pretrained Hierarchical Policy Tuning v1

## Purpose

This stage tunes a safer hierarchical decision policy for the full pretrained transfer-learning checkpoint.

The previous pretrained hierarchical policy achieved very strong known-test performance, but it accepted too many local unknown images. Therefore, this stage searches for stricter thresholds that can improve the local unknown manual-review rate while keeping useful known-test coverage.

## Model Context

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |
| Baseline B | pretrained transfer-learning model   |

## Config

This stage uses:

```text id="a9p0i1"
ml/configs/pretrained_safe_hierarchical_policy_tuning.yaml
```

The checkpoint is:

```text id="8fzc5u"
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

## Why Tuning Is Needed

The pretrained hierarchical policy v1 achieved:

| Metric                           |    Value |
| -------------------------------- | -------: |
| Known-test coverage              | 0.976562 |
| Known accepted reliability       | 0.957333 |
| Local unknown manual-review rate | 0.075000 |
| Local unknown acceptance rate    | 0.925000 |

This shows that the pretrained model is very reliable on known-test accepted decisions, but still too permissive for local unknown images.

## Search Space

The safe pretrained tuning searches stricter values for:

| Threshold                     | Purpose                                   |
| ----------------------------- | ----------------------------------------- |
| fine_confidence_threshold     | controls fine-label acceptance            |
| coarse_confidence_threshold   | controls coarse fallback acceptance       |
| coarse_margin_threshold       | requires separation between coarse groups |
| minimum_confidence_for_coarse | prevents weak coarse fallback decisions   |

The search includes stricter fine-confidence values such as:

```text id="95mf0v"
0.90, 0.95, 0.97, 0.99, 0.995
```

## Selection Goal

The policy should balance:

| Goal                               | Meaning                                    |
| ---------------------------------- | ------------------------------------------ |
| known-test coverage                | keep useful automatic decisions            |
| accepted-decision reliability      | keep accepted decisions accurate           |
| local unknown manual-review rate   | route more local unknown samples to review |
| local unknown acceptance reduction | avoid unsafe automatic acceptance          |

## Expected Outcome

The expected result is a safer pretrained hierarchical policy.

This tuned policy should be compared against:

| System                              | Purpose                                       |
| ----------------------------------- | --------------------------------------------- |
| scratch safe hierarchical policy    | previous safest scratch-trained policy        |
| pretrained hierarchical policy v1   | high known coverage but weak unknown handling |
| pretrained safe hierarchical policy | new tuned policy from this stage              |

## Research Meaning

This stage is important because it tests the main OpenWaste-HR safety trade-off using the stronger pretrained model.

If the tuned pretrained policy improves unknown manual-review while keeping strong known-test reliability, it becomes the best current candidate for the final OpenWaste-HR decision policy.
