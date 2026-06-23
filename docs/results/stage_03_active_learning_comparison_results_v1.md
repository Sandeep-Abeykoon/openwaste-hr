# Stage 3 Active Learning Comparison Results v1

## Purpose

This report compares the Stage 3 TrashNet + RealWaste + Garbage V2 baseline model against the Stage 3 active-learning retrained model.

The active-learning model adds 100 human-reviewed TrashBox known-class samples to the Stage 3 training set. The purpose is to check whether a small reviewed TrashBox batch improves adaptation to the next dataset source before fully adding TrashBox in Stage 4.

## Training Setup

### Stage 3 Baseline

- Training data: TrashNet + RealWaste + Garbage V2 known_train
- Known classes: cardboard, glass, metal, paper, plastic
- Model: MobileNetV3 Large pretrained
- Best validation macro-F1: 0.9432
- Best epoch: 13

### Stage 3 Active-Learning Model

- Training data: TrashNet + RealWaste + Garbage V2 known_train + 100 human-reviewed TrashBox additions
- Known classes: cardboard, glass, metal, paper, plastic
- Model: MobileNetV3 Large pretrained
- Best validation macro-F1: 0.9489
- Best epoch: 18

## Stage 3 Validation and Test Comparison

| Model | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Stage 3 baseline | 0.9430 | 0.9432 | 0.9445 | 0.9445 | 0.9437 |
| Stage 3 + active learning | 0.9485 | 0.9489 | 0.9508 | 0.9509 | 0.9510 |
| Difference | +0.0054 | +0.0058 | +0.0063 | +0.0064 | +0.0073 |

## External TrashBox Test Comparison

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Stage 3 baseline on TrashBox test | 0.6276 | 0.6292 | 0.6279 |
| Stage 3 + active learning on TrashBox test | 0.6869 | 0.6901 | 0.6873 |
| Difference | +0.0593 | +0.0609 | +0.0594 |

## Data Quality Note

During Stage 3 active-learning preparation, the TrashBox active-learning candidate pool contained 7,220 rows. Image validation found 7,218 valid images and 2 invalid/corrupted images. The invalid images were excluded from model scoring and retained separately for auditability.

The external TrashBox test evaluation used the cleaned TrashBox external test manifest.

## Interpretation

The Stage 3 active-learning cycle improved both the unchanged Stage 3 test split and the external TrashBox test split.

The improvement on the external TrashBox test set is especially important because TrashBox was not part of the Stage 3 baseline training data. The baseline model achieved 0.6292 macro-F1 on TrashBox, while the active-learning model achieved 0.6901 macro-F1 after adding only 100 human-reviewed TrashBox samples.

This demonstrates that active learning improved cross-dataset adaptation and did not harm the main Stage 3 known-class evaluation. In this stage, active learning produced a clear benefit on both internal and external evaluation.

## Research Finding

The Stage 3 active-learning cycle improved external TrashBox known-class macro-F1 from 0.6292 to 0.6901 using only 100 human-reviewed samples.

It also improved the unchanged Stage 3 test macro-F1 from 0.9445 to 0.9509. This supports the claim that small, targeted human-reviewed batches can improve robustness across dataset sources.

## Stage 3 Active Learning Status

The Stage 3 active-learning cycle is complete:

1. Trained Stage 3 TrashNet + RealWaste + Garbage V2 baseline.
2. Created TrashBox known-class active-learning candidate pool.
3. Validated candidate images and removed corrupted/unreadable files.
4. Scored TrashBox candidate samples.
5. Selected 100 active-learning review candidates.
6. Human-reviewed the selected samples.
7. Added only confirmed known-class samples to training.
8. Retrained the Stage 3 active-learning model.
9. Compared baseline vs active-learning model on unchanged Stage 3 test data and external TrashBox test data.

## Next Step

The next model stage is Stage 4:

- Train on TrashNet + RealWaste + Garbage V2 + TrashBox known classes.
- Use the cleaned manifest approach so corrupted TrashBox images are excluded.
- Evaluate final known-class performance.
- Then perform reject-option/open-set evaluation using textile and biological unknown samples.
