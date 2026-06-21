# Baseline TrashNet Evaluation v1

## Model

| Item | Value |
|---|---|
| Checkpoint | `ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt` |
| Model type | Closed-set forced-choice classifier |
| Dataset | TrashNet mapped into OpenWaste-HR taxonomy |
| Label level | Fine label |
| Number of classes | 5 |
| Classes | paper_cardboard, plastic, glass, metal, residual |

## Main Test Metrics

| metric | value |
| --- | --- |
| accuracy | 0.888021 |
| balanced_accuracy | 0.84305 |
| macro_f1 | 0.850962 |
| weighted_f1 | 0.887337 |

## Classification Report

| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| paper_cardboard | 0.947368 | 0.953642 | 0.950495 | 151.0 |
| plastic | 0.885714 | 0.849315 | 0.867133 | 73.0 |
| glass | 0.821429 | 0.907895 | 0.8625 | 76.0 |
| metal | 0.864407 | 0.822581 | 0.842975 | 62.0 |
| residual | 0.789474 | 0.681818 | 0.731707 | 22.0 |
| accuracy | 0.888021 | 0.888021 | 0.888021 | 0.888021 |
| macro avg | 0.861678 | 0.84305 | 0.850962 | 384.0 |
| weighted avg | 0.888281 | 0.888021 | 0.887337 | 384.0 |

## Confusion Matrix

![Baseline confusion matrix](figures/pretrained_trashnet_confusion_matrix_v1.png)

## Research Interpretation

This is the first closed-set baseline. It always predicts one of the known TrashNet-derived fine labels.

This result should not be treated as the final OpenWaste-HR contribution. It is the comparison point for later experiments involving confidence-based rejection, unknown detection, hierarchical coarse fallback, and local active learning.

Important limitation: this TrashNet baseline does not include organic or e-waste/battery classes, and it does not test unknown rejection.
