# Pretrained Baseline Evaluation Summary v1

## Purpose

This stage evaluates the pretrained transfer-learning model as Baseline B.

## Model Context

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |
| Baseline B | pretrained transfer-learning model   |

## Evaluation Config

```text id="bucarb"
ml/configs/evaluate_pretrained_trashnet.yaml
```

Expected checkpoint:

```text id="8m3gh3"
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

## Current Comparison Before Separate Evaluation

| Model                                  | Test Accuracy | Test Macro-F1 |
| -------------------------------------- | ------------: | ------------: |
| Baseline A: scratch-trained            |      0.692708 |      0.645600 |
| Baseline B: pretrained training result |      0.888000 |      0.851000 |

## Why This Matters

The pretrained model appears much stronger than the scratch-trained model on known-test classification.

The next stage checks this through the formal evaluation script and prepares the pretrained checkpoint for downstream reject-option, local unknown, and hierarchical decision evaluation.

## Next Evaluation After This

After this closed-set pretrained evaluation, the next stages should evaluate the pretrained checkpoint using:

1. confidence reject baseline
2. max-logit and energy baselines
3. local unknown evaluation
4. hierarchical decision policy
5. safe hierarchical tuning
6. final comparison with the original scratch-trained pipeline
