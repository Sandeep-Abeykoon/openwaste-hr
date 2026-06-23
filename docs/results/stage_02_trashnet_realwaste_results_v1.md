# Stage 2 TrashNet + RealWaste Results v1

## Experiment

- Experiment name: stage_02_trashnet_realwaste
- Model: MobileNetV3 Large pretrained backbone
- Known classes: cardboard, glass, metal, paper, plastic
- Training datasets: TrashNet + RealWaste known classes
- Unknown evaluation classes: textile and biological
- Training protocol: closed-set 5-class dataset expansion
- Early stopping monitor: validation macro-F1

## Training Summary

- Best epoch: 26
- Best validation macro-F1: 0.9455
- Unknown validation prediction count: 1,660
- Unknown test prediction count: 1,660

## Validation Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9447 |
| Macro-F1 | 0.9455 |
| Balanced accuracy | 0.9437 |
| Loss | 0.2123 |

## Test Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9432 |
| Macro-F1 | 0.9447 |
| Balanced accuracy | 0.9445 |
| Loss | 0.2913 |

## Interpretation

The Stage 2 model trained on TrashNet and RealWaste achieved strong known-class performance. Compared with the Stage 1 TrashNet-only baseline, this confirms that adding RealWaste known-class data improves the model’s ability to handle a broader public dataset distribution.

This stage is important because Stage 1 showed clear domain shift between TrashNet and RealWaste. Stage 2 addresses that shift by training on both datasets directly while still keeping Food Organics and Textile Trash outside training for unknown/reject-option evaluation.

## Next Required Step

Before moving to Garbage Classification V2, the Stage 2 active-learning cycle must be completed.

The Stage 2 active-learning candidate pool should come from Garbage Classification V2 known-class training samples. This checks how well the TrashNet + RealWaste model handles the next unseen dataset source before fully adding it into training.

The next active-learning cycle will:

1. Score Garbage Classification V2 known-class candidate samples.
2. Select uncertain, low-margin, and confident-wrong samples.
3. Create a human-review sheet.
4. Add only human-confirmed known-class samples to active-learning retraining.
5. Retrain and compare improvement before Stage 3.
