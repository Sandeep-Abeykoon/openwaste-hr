# Expanded Public Pretrained Training Preparation v1

## Purpose

This stage prepares the training configuration for Baseline C.

Baseline C is the pretrained expanded public dataset model.

## Baseline C Definition

```text
Baseline C = pretrained expanded public dataset model
```

This model uses the combined TrashNet-style and RealWaste known training data.

## Input Manifests

| Split      | File                                              |
| ---------- | ------------------------------------------------- |
| train      | ml/data/splits/expanded_public_known_train_v1.csv |
| validation | ml/data/splits/expanded_public_known_val_v1.csv   |
| test       | ml/data/splits/expanded_public_known_test_v1.csv  |

## Dataset Scale

| Split                     | Count |
| ------------------------- | ----: |
| expanded known train      |  4869 |
| expanded known validation |  1042 |
| expanded known test       |  1050 |

## Why This Stage Is Needed

The previous strongest model was trained mainly using the TrashNet-style dataset. That was useful for the first complete OpenWaste-HR prototype, but it was still limited by the size and diversity of the data.

The expanded public dataset adds RealWaste known samples and introduces stronger coverage for the `organic` class.

## Training Plan

The training plan is:

1. create expanded pretrained config
2. create 1-epoch smoke config
3. run smoke training
4. confirm that the combined dataset loads correctly
5. confirm that the model can train and evaluate on the expanded public splits
6. only then run full expanded public pretrained training

## Output Names

| Config         | Output Prefix                       |
| -------------- | ----------------------------------- |
| full training  | expanded_public_pretrained_v1       |
| smoke training | expanded_public_pretrained_smoke_v1 |

## Research Meaning

This stage moves OpenWaste-HR beyond TrashNet-only training.

If the expanded public pretrained model improves over the TrashNet pretrained model, the thesis can show that the project became stronger through dataset expansion rather than relying only on a small baseline dataset.
