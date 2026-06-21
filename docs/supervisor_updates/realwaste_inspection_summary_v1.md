# RealWaste Inspection Summary v1

## Purpose

This stage inspects the actual RealWaste manifest after dataset intake.

## Input

```text
ml/data/splits/realwaste_manifest_v1.csv
```

## Expected Summary

| Metric                       | Value |
| ---------------------------- | ----: |
| total samples                |  4752 |
| known samples                |  4434 |
| unknown/future-class samples |   318 |

## Important Mapping Decision

The RealWaste `Textile Trash` class is treated as:

```text
unknown / future_class_candidate
```

It is not forced into residual or another current known class.

## Why This Matters

This keeps the expanded dataset aligned with the OpenWaste-HR open-set design.

RealWaste now provides:

1. additional known-class training and evaluation data
2. public-dataset unknown/future-class samples for open-set evaluation

## Next Stage

After this inspection, the next step should create combined TrashNet + RealWaste training manifests.
