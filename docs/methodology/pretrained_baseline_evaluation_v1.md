# Pretrained Baseline Evaluation v1

## Purpose

This stage evaluates the full pretrained transfer-learning model on the known test set.

The scratch-trained model is treated as:

```text id="r4a4qs"
Baseline A = scratch-trained TrashNet-style model
```

The pretrained model is treated as:

```text id="57buz9"
Baseline B = pretrained transfer-learning model
```

The purpose of this stage is to produce a clean closed-set evaluation report for Baseline B before running reject-option, local unknown, and hierarchical policy evaluation.

## Evaluation Config

The evaluation config is:

```text id="sw5blu"
ml/configs/evaluate_pretrained_trashnet.yaml
```

The pretrained checkpoint should be:

```text id="falx6p"
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

## Why This Evaluation Is Needed

The training script already printed a test accuracy and macro-F1 value after training. However, a separate evaluation stage is still useful because it creates a structured report, predictions file, classification metrics, and figures using the same evaluation workflow as the original scratch-trained baseline.

This makes the comparison between Baseline A and Baseline B cleaner.

## Expected Comparison Direction

The original scratch-trained baseline result was:

| Model                       | Test Accuracy | Test Macro-F1 |
| --------------------------- | ------------: | ------------: |
| Baseline A: scratch-trained |      0.692708 |      0.645600 |

The pretrained training run reported:

| Model                  | Test Accuracy | Test Macro-F1 |
| ---------------------- | ------------: | ------------: |
| Baseline B: pretrained |      0.888000 |      0.851000 |

The separate evaluation report will confirm and record the pretrained model result using the evaluation pipeline.

## Evaluation Outputs

This stage is expected to produce separate pretrained evaluation outputs such as:

| Output Type             | Purpose                                     |
| ----------------------- | ------------------------------------------- |
| predictions CSV         | stores true and predicted labels            |
| metrics JSON            | stores evaluation metrics                   |
| classification report   | records per-class precision, recall, and F1 |
| confusion matrix figure | visualises class-level errors               |
| results Markdown report | thesis-ready evaluation summary             |

## Research Meaning

This stage begins the formal comparison between the scratch-trained and pretrained baselines.

If the pretrained evaluation confirms the training result, it supports the argument that pretrained visual features improve known-class waste recognition. The next important question is whether this improvement also improves reject-option behaviour, local unknown handling, and hierarchical decision safety.
