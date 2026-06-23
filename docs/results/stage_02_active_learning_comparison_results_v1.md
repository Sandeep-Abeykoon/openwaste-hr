# Stage 2 Active Learning Comparison Results v1

## Purpose

This report compares the Stage 2 TrashNet + RealWaste baseline model against the Stage 2 active-learning retrained model.

The active-learning model adds 100 human-reviewed Garbage V2 known-class samples to the Stage 2 training set. The purpose is to check whether a small reviewed batch improves adaptation to the next dataset source before fully adding Garbage V2 in Stage 3.

## Training Setup

### Stage 2 Baseline

- Training data: TrashNet + RealWaste known_train
- Known classes: cardboard, glass, metal, paper, plastic
- Model: MobileNetV3 Large pretrained
- Best validation macro-F1: 0.9455
- Best epoch: 26

### Stage 2 Active-Learning Model

- Training data: TrashNet + RealWaste known_train + 100 human-reviewed Garbage V2 additions
- Known classes: cardboard, glass, metal, paper, plastic
- Model: MobileNetV3 Large pretrained
- Best validation macro-F1: 0.9538
- Best epoch: 28

## Stage 2 Validation and Test Comparison

| Model | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Test Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Stage 2 baseline | 0.9447 | 0.9455 | 0.9432 | 0.9447 | 0.9445 |
| Stage 2 + active learning | 0.9535 | 0.9538 | 0.9347 | 0.9350 | 0.9373 |
| Difference | +0.0088 | +0.0083 | -0.0085 | -0.0097 | -0.0072 |

## External Garbage V2 Test Comparison

| Model | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Stage 2 baseline on Garbage V2 test | 0.7477 | 0.7494 | 0.7543 |
| Stage 2 + active learning on Garbage V2 test | 0.8549 | 0.8540 | 0.8547 |
| Difference | +0.1072 | +0.1046 | +0.1004 |

## Interpretation

The Stage 2 active-learning cycle improved validation performance and substantially improved external Garbage V2 test performance.

The unchanged Stage 2 test performance reduced slightly after adding the 100 reviewed Garbage V2 samples. However, this reduction was small compared with the large improvement on the next external dataset source.

This result shows that the active-learning update helped the model adapt to Garbage V2 before fully adding Garbage V2 into the main training set. Therefore, the active-learning cycle is useful for cross-dataset/domain adaptation rather than simply improving the original closed-set test split.

## Research Finding

The Stage 2 active-learning cycle improved Garbage V2 external known-class macro-F1 from 0.7494 to 0.8540 using only 100 human-reviewed samples.

This supports the OpenWaste-HR claim that active learning can efficiently improve cross-dataset robustness with a small amount of human labelling.

## Stage 2 Active Learning Status

The Stage 2 active-learning cycle is complete:

1. Trained Stage 2 TrashNet + RealWaste baseline.
2. Scored Garbage V2 known-class candidate pool.
3. Selected 100 active-learning review candidates.
4. Human-reviewed the selected samples.
5. Added only confirmed known-class samples to training.
6. Retrained the Stage 2 active-learning model.
7. Compared baseline vs active-learning model on unchanged Stage 2 test data and external Garbage V2 test data.

## Next Step

The next model stage is Stage 3:

- Train on TrashNet + RealWaste + Garbage V2 known classes.
- Keep textile and biological samples outside known-class training for unknown/reject-option evaluation.
- Evaluate Stage 3 known-class performance.
- Complete a Stage 3 active-learning cycle before moving to TrashBox.
