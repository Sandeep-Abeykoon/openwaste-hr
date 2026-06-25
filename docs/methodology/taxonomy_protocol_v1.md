# OpenWaste-HR Clean 5-Class Taxonomy Protocol v1

## Purpose

This protocol defines the clean restarted taxonomy for OpenWaste-HR. The project no longer uses residual, trash, or miscellaneous classes as known classes because those labels contain multiple waste types and can confuse both classification and unknown-detection evaluation.

The project will use a strict 5-class known taxonomy and separate held-out unknown classes for reject-option evaluation and active learning.

## Known Classes

The known training, validation, and known-test classes are:

- cardboard
- glass
- metal
- paper
- plastic

Only clear single-label images from these classes may be used in known-class model training.

## Unknown Evaluation Classes

The public unknown evaluation classes are:

- textile / clothes / cloth / fabric
- biological / food organic / food organics

These classes must not be used during known-class training. They are held out to test whether the model can reject unseen waste types as unknown or manual review.

## Excluded Classes

The following classes are excluded from the main clean protocol:

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

These classes may only be used later as optional stress-test unknown data if the methodology explicitly states that they were not used in training.

## Dataset Harmonisation Rule

All public datasets must be mapped into the same canonical taxonomy before training. Dataset-specific labels are mapped using:

`ml/configs/source_label_mapping.csv`

The canonical training labels are only:

`cardboard`, `glass`, `metal`, `paper`, and `plastic`.

## Active Learning Rule

After every model stage, an active learning cycle must be completed before moving to the next model stage.

The active learning cycle is:

1. Train the current model.
2. Run inference on validation, unknown, and difficult/local samples.
3. Select uncertain, rejected, or confusing samples.
4. Human-review selected samples.
5. Add only human-confirmed known-class samples back into the training pool.
6. Keep true unknown samples in unknown evaluation or manual-review records.
7. Retrain and compare improvement.

True unknown samples must not be trained as a sixth class.

## Evaluation Rule

Each model stage must report both closed-set and reject-option metrics.

Closed-set metrics:

- accuracy
- macro-F1
- balanced accuracy
- confusion matrix

Reject-option / open-set metrics:

- unknown rejection rate
- AUROC for known-vs-unknown separation
- ECE
- coverage-risk curve

## Dataset Expansion Order

The project will be evaluated in a logical dataset-by-dataset manner:

1. TrashNet / Garbage Classification 5-class baseline.
2. Add RealWaste known classes.
3. Add Garbage Dataset V2 known classes and use clothes/biological as unknown evaluation.
4. Add TrashBox known classes.
5. Train and evaluate final combined public dataset.
6. Apply reject-option/open-set scoring and active learning updates.

This design is used to show how each dataset expansion affects performance, robustness, and unknown rejection.
