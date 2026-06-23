# Stage 4 Final Combined Model Results v1

## Experiment

- Experiment name: stage_04_add_trashbox_clean_v1
- Model: MobileNetV3 Large pretrained backbone
- Known classes: cardboard, glass, metal, paper, plastic
- Training datasets: TrashNet + RealWaste + Garbage V2 + TrashBox
- Unknown evaluation classes: textile and biological
- Training protocol: final combined closed-set 5-class model
- Early stopping monitor: validation macro-F1
- Clean split: corrupted TrashBox images removed before training

## Data Quality Handling

The original Stage 4 training split contained 15,960 known-training rows. Image validation found 2 corrupted/unreadable TrashBox images. These were removed before training.

Final clean Stage 4 split:

| Split | Rows |
|---|---:|
| known_train | 15,958 |
| known_val | 3,417 |
| known_test | 3,426 |
| unknown_val | 1,660 |
| unknown_test | 1,660 |

## Training Summary

| Item | Value |
|---|---:|
| Best epoch | 10 |
| Best validation macro-F1 | 0.9237 |
| Unknown validation prediction count | 1,660 |
| Unknown test prediction count | 1,660 |

## Overall Validation Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9230 |
| Macro-F1 | 0.9237 |
| Balanced accuracy | 0.9244 |
| Loss | 0.2976 |

## Overall Test Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9212 |
| Macro-F1 | 0.9213 |
| Balanced accuracy | 0.9217 |
| Loss | 0.3070 |

## Per-Dataset Test Evaluation

| Dataset | Test rows | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| TrashNet | 355 | 0.9634 | 0.9636 | 0.9639 |
| RealWaste | 472 | 0.9089 | 0.9111 | 0.9101 |
| Garbage V2 | 1,082 | 0.9538 | 0.9532 | 0.9551 |
| TrashBox | 1,517 | 0.8919 | 0.8916 | 0.8915 |
| Overall | 3,426 | 0.9212 | 0.9213 | 0.9217 |

## Interpretation

The Stage 4 final model achieved strong known-class performance across all four public dataset sources. The overall test macro-F1 was 0.9213.

Per-dataset evaluation shows that TrashNet and Garbage V2 achieved the highest performance, while RealWaste and TrashBox were more challenging. TrashBox remained the hardest dataset source, but the final combined training significantly improved performance compared with the earlier external TrashBox evaluation.

TrashBox macro-F1 improved from 0.6292 with the Stage 3 baseline model to 0.6901 after Stage 3 active learning, and then to 0.8916 after full Stage 4 combined training.

This confirms that staged dataset expansion improved cross-dataset robustness while keeping the taxonomy fixed to five clean known classes.

## Comparison Across Main Stages

| Stage | Training data | Test macro-F1 |
|---|---|---:|
| Stage 1 baseline | TrashNet | 0.9320 |
| Stage 1 + active learning | TrashNet + 100 reviewed RealWaste samples | 0.9268 |
| Stage 2 baseline | TrashNet + RealWaste | 0.9447 |
| Stage 2 + active learning | TrashNet + RealWaste + 100 reviewed Garbage V2 samples | 0.9350 |
| Stage 3 baseline | TrashNet + RealWaste + Garbage V2 | 0.9445 |
| Stage 3 + active learning | TrashNet + RealWaste + Garbage V2 + 100 reviewed TrashBox samples | 0.9509 |
| Stage 4 final combined | TrashNet + RealWaste + Garbage V2 + TrashBox | 0.9213 |

## Research Finding

The final combined model demonstrates that a clean five-class taxonomy can be trained across multiple public waste datasets with strong known-class performance.

The staged expansion process also showed that active learning helped the model adapt to new dataset sources before full dataset integration. This supports the OpenWaste-HR research design: dataset harmonisation, staged expansion, human-reviewed active learning, and final reject-option evaluation.

## Stage 4 Status

The Stage 4 final known-class model is complete:

1. Created clean Stage 4 split.
2. Removed corrupted TrashBox images.
3. Trained final combined model.
4. Evaluated aggregate known-class performance.
5. Evaluated per-dataset known-class performance.

## Next Step

The next step is reject-option/open-set evaluation.

The model should now be evaluated on:

- known validation and known test samples
- unknown validation and unknown test samples
- unknown classes: textile and biological

The reject-option evaluation should compare confidence thresholding, temperature-scaled confidence, max-logit scoring, and energy scoring.
