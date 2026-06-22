# Expanded Public Pretrained Training v1

## Purpose

This stage runs the full Baseline C training experiment.

Baseline C is the pretrained expanded public dataset model.

## Baseline C Definition

```text id="0ks4aq"
Baseline C = pretrained expanded public dataset model
```

## Training Data

This model uses the combined known training data from:

| Dataset                | Role                           |
| ---------------------- | ------------------------------ |
| TrashNet-style dataset | original baseline source       |
| RealWaste              | expanded public dataset source |

## Input Manifests

| Split      | File                                              |
| ---------- | ------------------------------------------------- |
| train      | ml/data/splits/expanded_public_known_train_v1.csv |
| validation | ml/data/splits/expanded_public_known_val_v1.csv   |
| test       | ml/data/splits/expanded_public_known_test_v1.csv  |

## Split Sizes

| Split      | Count |
| ---------- | ----: |
| train      |  4869 |
| validation |  1042 |
| test       |  1050 |

## Known Training Classes

The expanded public model is trained on six known fine labels:

| Fine Label      |
| --------------- |
| paper_cardboard |
| plastic         |
| glass           |
| metal           |
| organic         |
| residual        |

The `unknown` label is not used as a training class.

## Why Unknown Is Not Trained

The RealWaste `Textile Trash` class is preserved separately as public unknown/future-class data.

This supports the OpenWaste-HR open-set design because outside-taxonomy objects should not be forced into known training classes.

## Expected Output

The full training run creates:

| Output           | Path                                                                                                  |
| ---------------- | ----------------------------------------------------------------------------------------------------- |
| best checkpoint  | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_best.pt            |
| final checkpoint | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_final.pt           |
| class mapping    | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_class_mapping.json |
| training metrics | ml/outputs/metrics/expanded_public_pretrained_v1_training_metrics.csv                                 |
| test metrics     | ml/outputs/metrics/expanded_public_pretrained_v1_test_metrics.json                                    |

## Research Meaning

This stage tests whether dataset expansion improves OpenWaste-HR beyond the TrashNet-only pretrained model.

The result should later be compared against:

| Model                            | Purpose                              |
| -------------------------------- | ------------------------------------ |
| Scratch TrashNet baseline        | weak baseline                        |
| Pretrained TrashNet baseline     | previous strongest known-class model |
| Expanded public pretrained model | tests RealWaste dataset expansion    |

## Next Stage

After full training, the next stage should evaluate Baseline C using the standard closed-set evaluation script.
