# Pretrained Reject-Option Evaluation Summary v1

## Purpose

This stage evaluates reject-option methods using the pretrained transfer-learning checkpoint.

## Model Context

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |
| Baseline B | pretrained transfer-learning model   |

## Checkpoint

```text id="391xeg"
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

## Configs

| Config                                     | Purpose                        |
| ------------------------------------------ | ------------------------------ |
| confidence_reject_pretrained_trashnet.yaml | confidence-threshold rejection |
| open_set_score_pretrained_trashnet.yaml    | max-logit and energy rejection |

## What This Stage Tests

This stage checks whether the pretrained model improves:

* known-test selective accuracy
* known-test selective macro-F1
* accepted prediction reliability
* reject-option behaviour before local unknown testing

## Next Stage

After this, the pretrained checkpoint should be evaluated on the local unknown dataset to check unknown rejection and false acceptance.
