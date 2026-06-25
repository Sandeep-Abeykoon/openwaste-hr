# OpenWaste-HR Dataset Intake Guide v1

## Purpose

This guide defines how public datasets must be placed locally before manifest generation.

Raw dataset images are not committed to GitHub. Only configs, manifest CSV files, methodology documents, and result summaries are committed.

## Active Public Datasets

The clean restart uses four public dataset sources:

1. TrashNet
2. RealWaste
3. Garbage Classification V2
4. TrashBox

The previous Kaggle Garbage Classification dataset is omitted because it overlaps with the TrashNet-style six-class structure and would add duplication rather than a clearer research contribution.

## Local Raw Dataset Folders

Place datasets under:

ml/data/raw/trashnet/
ml/data/raw/realwaste/
ml/data/raw/garbage_v2/
ml/data/raw/trashbox/

## Dataset Roles

| Dataset | Main role |
|---|---|
| TrashNet | Initial 5-class baseline |
| RealWaste | Real-world dataset expansion |
| Garbage Classification V2 | Public expansion with known classes and unknown classes |
| TrashBox | Additional public expansion and external variation |

## Known Training Labels

Only these labels are allowed for training, validation, and known testing:

- cardboard
- glass
- metal
- paper
- plastic

## Unknown Evaluation Labels

These labels are held out from training and used only for unknown/reject-option evaluation:

- textile / clothes / cloth / fabric
- biological / food organic / food organics

## Excluded Labels

These labels are excluded from the main clean protocol:

- residual
- trash
- miscellaneous
- mixed
- unclear
- battery
- e-waste
- medical
- shoes
- vegetation

They must not be used as known training classes.

## Required Rule Before Manifest Generation

Before generating any manifest, each dataset must be checked against:

ml/configs/source_label_mapping.csv

Any source folder that does not map to a known, unknown, or excluded canonical label must be reported as an unmapped label.

## Active Learning Rule

After every model stage, active learning must be completed before moving to the next stage.

For each stage:

1. Train the current model.
2. Run inference on validation, unknown, and difficult samples.
3. Select uncertain, rejected, and confusing samples.
4. Human-review those samples.
5. Add only human-confirmed known-class samples to the next training pool.
6. Keep true unknown samples outside training.
7. Retrain and compare improvement.

## Dataset Expansion Order

The planned order is:

1. TrashNet 5-class baseline.
2. TrashNet + RealWaste known classes.
3. Add Garbage Classification V2 known classes and use clothes/biological as public unknowns.
4. Add TrashBox known classes.
5. Final combined public dataset.
6. Reject-option and open-set evaluation.
7. Active-learning updated final model.
