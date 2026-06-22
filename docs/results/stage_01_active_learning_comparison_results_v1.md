# Stage 1 Active Learning Comparison Results v1

## Purpose

This report compares the original Stage 1 TrashNet-only baseline against the Stage 1 active-learning updated model.

The goal is to check whether a small human-reviewed RealWaste active-learning batch improves performance on RealWaste domain-shift data while preserving strong TrashNet performance.

## Training Setup

### Stage 1 Baseline

- Training data: TrashNet known_train only
- Known classes: cardboard, glass, metal, paper, plastic
- Model: MobileNetV3 Large pretrained
- Best validation macro-F1: 0.9398
- Best epoch: 19

### Stage 1 Active-Learning Model

- Training data: TrashNet known_train + 100 human-reviewed RealWaste known-class additions
- Known classes: cardboard, glass, metal, paper, plastic
- Model: MobileNetV3 Large pretrained
- Best validation macro-F1: 0.9476
- Best epoch: 17

## TrashNet Test Comparison

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Stage 1 baseline | 0.9324 | 0.9320 | 0.9357 |
| Stage 1 + active learning | 0.9268 | 0.9268 | 0.9308 |
| Difference | -0.0056 | -0.0052 | -0.0049 |

## RealWaste Test Comparison

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Stage 1 baseline on RealWaste test | 0.5191 | 0.5161 | 0.5073 |
| Stage 1 + active learning on RealWaste test | 0.7627 | 0.7606 | 0.7616 |
| Difference | +0.2436 | +0.2445 | +0.2542 |

## Interpretation

The Stage 1 TrashNet-only baseline achieved strong performance on TrashNet test data, but it performed poorly on RealWaste test data. This shows clear dataset/domain shift between TrashNet and RealWaste.

After the active-learning cycle, only 100 human-reviewed RealWaste known-class samples were added to training. This substantially improved RealWaste test performance:

- RealWaste accuracy improved from 0.5191 to 0.7627.
- RealWaste macro-F1 improved from 0.5161 to 0.7606.
- RealWaste balanced accuracy improved from 0.5073 to 0.7616.

The original TrashNet test performance dropped only slightly, from 0.9320 macro-F1 to 0.9268 macro-F1. This is an acceptable trade-off because the active-learning model became much more robust to RealWaste domain-shift data while largely preserving original benchmark performance.

## Research Finding

This result supports the OpenWaste-HR research claim that a small human-reviewed active-learning batch can improve domain adaptation for waste classification. It also shows why the project should not rely only on closed-set benchmark accuracy.

## Stage 1 Active Learning Status

The Stage 1 active-learning cycle is complete:

1. Trained Stage 1 TrashNet baseline.
2. Scored RealWaste known-class candidate pool.
3. Selected 100 confident-wrong active-learning candidates.
4. Human-reviewed the selected samples.
5. Added only confirmed known-class samples to training.
6. Retrained the Stage 1 active-learning model.
7. Compared baseline vs active-learning model on TrashNet and RealWaste test data.

## Next Step

The next model stage is Stage 2:

- Train on TrashNet + RealWaste known classes.
- Evaluate known-class performance.
- Run reject-option/open-set scoring on textile and biological unknown data.
- Complete another active-learning cycle before moving to the next dataset expansion.
