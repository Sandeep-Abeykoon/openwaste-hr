# Pretrained Hierarchical Policy Evaluation Summary v1

## Purpose

This stage evaluates the hierarchical decision policy using the pretrained transfer-learning checkpoint.

## Model Context

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |
| Baseline B | pretrained transfer-learning model   |

## Config

```text id="v3qq7x"
ml/configs/pretrained_hierarchical_decision_policy.yaml
```

Checkpoint:

```text id="xyvqzj"
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

## Decision Types

| Decision Type | Meaning                               |
| ------------- | ------------------------------------- |
| fine_label    | detailed known label                  |
| coarse_label  | broad fallback category               |
| manual_review | uncertain case routed to human review |

## Current Threshold Focus

The pretrained hierarchical policy uses a stricter fine-label threshold:

```text id="dgovph"
fine_confidence_threshold = 0.9900
```

## What This Stage Checks

This stage checks whether the pretrained checkpoint improves:

* known-test hierarchical decision reliability
* fine_label decision quality
* coarse_label fallback usefulness
* manual_review routing
* local unknown handling

## Next Stage

After this, the next stage should tune the pretrained hierarchical policy for a safer coverage/unknown-handling trade-off.
