# RealWaste Inspection v1

## Purpose

This stage inspects the actual RealWaste manifest after dataset intake.

The goal is to verify that the RealWaste dataset was placed correctly, mapped safely into the OpenWaste-HR taxonomy, and prepared for the next expanded training phase.

## Input Manifest

This stage uses:

```text
ml/data/splits/realwaste_manifest_v1.csv
```

## Expected Dataset Summary

The RealWaste manifest contains:

| Metric                       | Value |
| ---------------------------- | ----: |
| total samples                |  4752 |
| known samples                |  4434 |
| unknown/future-class samples |   318 |
| known train                  |  3103 |
| known validation             |   665 |
| known test                   |   666 |
| unknown test                 |   318 |

## Unknown/Future-Class Split

The unknown/future-class split comes from the RealWaste label:

```text
Textile Trash
```

This label is intentionally mapped as:

| Field        | Value                  |
| ------------ | ---------------------- |
| fine_label   | unknown                |
| coarse_label | unknown                |
| is_known     | false                  |
| usage        | unknown_test           |
| mapping_role | future_class_candidate |

This mapping preserves the OpenWaste-HR open-set design because Textile Trash is outside the current known fine-label taxonomy.

## Inspection Checks

This stage checks:

| Check                     | Purpose                                         |
| ------------------------- | ----------------------------------------------- |
| total row count           | confirms manifest size                          |
| usage distribution        | confirms train/validation/test/unknown split    |
| fine-label distribution   | checks class balance                            |
| coarse-label distribution | checks hierarchy balance                        |
| mapping-role distribution | confirms known and unknown roles                |
| image path existence      | confirms dataset paths are valid                |
| image readability sample  | checks whether sample image files can be opened |

## Research Meaning

This inspection stage is important because expanded training should not begin until the dataset is verified.

It also documents the key research decision that RealWaste contributes both known training data and a public-dataset unknown/future-class split.

## Next Stage

After inspection, the next stage should prepare combined TrashNet + RealWaste training manifests for Baseline C.
