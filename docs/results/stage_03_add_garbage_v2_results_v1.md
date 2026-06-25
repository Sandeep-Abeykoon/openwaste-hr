# Stage 3 TrashNet + RealWaste + Garbage V2 Results v1

## Experiment

- Experiment name: stage_03_add_garbage_v2
- Model: MobileNetV3 Large pretrained backbone
- Known classes: cardboard, glass, metal, paper, plastic
- Training datasets: TrashNet + RealWaste + Garbage V2 known classes
- Unknown evaluation classes: textile and biological
- Training protocol: closed-set 5-class dataset expansion
- Early stopping monitor: validation macro-F1

## Training Summary

- Best epoch: 13
- Best validation macro-F1: 0.9432
- Early stopping epoch: 19
- Unknown validation prediction count: 1,660
- Unknown test prediction count: 1,660

## Validation Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9430 |
| Macro-F1 | 0.9432 |
| Balanced accuracy | 0.9433 |
| Loss | 0.2320 |

## Test Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9445 |
| Macro-F1 | 0.9445 |
| Balanced accuracy | 0.9437 |
| Loss | 0.1751 |

## Interpretation

The Stage 3 model trained on TrashNet, RealWaste, and Garbage V2 achieved strong known-class performance on the expanded public dataset split.

The final test macro-F1 of 0.9445 shows that adding Garbage V2 did not reduce overall known-class classification quality. The model remained stable after expanding from two datasets to three datasets.

This stage is important because Garbage V2 provides a larger and more diverse set of known-class samples, while textile/clothes and biological/food-organic samples remain excluded from known-class training and reserved for reject-option/open-set evaluation.

## Comparison with Earlier Stages

| Stage | Training data | Test macro-F1 |
|---|---|---:|
| Stage 1 baseline | TrashNet | 0.9320 |
| Stage 1 + active learning | TrashNet + 100 reviewed RealWaste samples | 0.9268 |
| Stage 2 baseline | TrashNet + RealWaste | 0.9447 |
| Stage 2 + active learning | TrashNet + RealWaste + 100 reviewed Garbage V2 samples | 0.9350 |
| Stage 3 baseline | TrashNet + RealWaste + Garbage V2 | 0.9445 |

## Research Finding

The Stage 3 result confirms that dataset expansion can maintain strong known-class performance while increasing dataset diversity. This supports the staged research design, where each new dataset source is added only after baseline evaluation and active-learning analysis.

## Stage 3 Status

The Stage 3 baseline model is complete:

1. Trained on TrashNet + RealWaste + Garbage V2 known classes.
2. Evaluated on the Stage 3 validation and test splits.
3. Generated predictions for unknown validation and unknown test samples.
4. Preserved textile and biological samples for reject-option evaluation.

## Next Required Step

Before moving to Stage 4, the Stage 3 active-learning cycle must be completed.

The Stage 3 active-learning candidate pool should come from TrashBox known-class training samples. This checks how well the Stage 3 model handles the next unseen dataset source before fully adding TrashBox into Stage 4 training.
