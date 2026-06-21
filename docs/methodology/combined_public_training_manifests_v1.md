# Combined Public Training Manifests v1

## Purpose

This stage creates the expanded public dataset manifests for OpenWaste-HR.

The current trained model is based mainly on the TrashNet-style dataset. RealWaste has now been added and inspected. This stage combines the known training, validation, and test samples from both datasets.

## Input Datasets

| Dataset                | Role                      |
| ---------------------- | ------------------------- |
| TrashNet-style dataset | original baseline dataset |
| RealWaste              | expanded public dataset   |

## Combined Dataset Purpose

The combined dataset prepares Baseline C:

```text
Baseline C = pretrained expanded public dataset model
```

The goal is to test whether using more public waste data improves:

* known-class accuracy
* macro-F1
* class balance
* hierarchical policy quality
* local unknown handling
* public unknown handling

## Input Manifest Files

| Input                    | File                                         |
| ------------------------ | -------------------------------------------- |
| TrashNet train           | ml/data/splits/known_train.csv               |
| TrashNet validation      | ml/data/splits/known_val.csv                 |
| TrashNet test            | ml/data/splits/known_test.csv                |
| RealWaste train          | ml/data/splits/realwaste_known_train_v1.csv  |
| RealWaste validation     | ml/data/splits/realwaste_known_val_v1.csv    |
| RealWaste test           | ml/data/splits/realwaste_known_test_v1.csv   |
| RealWaste public unknown | ml/data/splits/realwaste_unknown_test_v1.csv |

## Output Manifest Files

| Output                                   | Purpose                                       |
| ---------------------------------------- | --------------------------------------------- |
| expanded_public_manifest_v1.csv          | combined known and public unknown manifest    |
| expanded_public_known_train_v1.csv       | TrashNet + RealWaste known training samples   |
| expanded_public_known_val_v1.csv         | TrashNet + RealWaste known validation samples |
| expanded_public_known_test_v1.csv        | TrashNet + RealWaste known test samples       |
| expanded_public_unknown_test_v1.csv      | RealWaste public unknown/future-class samples |
| expanded_public_manifest_summary_v1.json | summary of counts and label distributions     |

## Expected Combined Counts

| Split                        | Expected Count |
| ---------------------------- | -------------: |
| expanded known train         |           4869 |
| expanded known validation    |           1042 |
| expanded known test          |           1050 |
| expanded public unknown test |            318 |
| total combined manifest      |           7279 |

## Label Expansion

The expanded public training data adds stronger support for existing known classes and introduces `organic` into the training set through RealWaste.

The current known fine labels in the expanded training data are expected to include:

| Fine Label      |
| --------------- |
| paper_cardboard |
| plastic         |
| glass           |
| metal           |
| organic         |
| residual        |

The `e_waste_battery` class remains part of the OpenWaste-HR taxonomy, but it is not yet represented in the current public training data.

## Unknown/Future-Class Handling

RealWaste `Textile Trash` remains separate as:

| Field        | Value                  |
| ------------ | ---------------------- |
| fine_label   | unknown                |
| coarse_label | unknown                |
| is_known     | false                  |
| usage        | unknown_test           |
| mapping_role | future_class_candidate |

This keeps the expanded dataset aligned with the OpenWaste-HR open-set design.

## Research Meaning

This stage prepares the first stronger dataset-based improvement over the TrashNet-only model.

After this stage, the next step is to configure and train a pretrained model using the expanded public training manifest.
