# Combined Public Training Manifests Summary v1

## Purpose

This stage creates the combined public training manifests for the next OpenWaste-HR model.

## Combined Sources

| Dataset                | Role                                           |
| ---------------------- | ---------------------------------------------- |
| TrashNet-style dataset | original baseline training source              |
| RealWaste              | expanded public training and evaluation source |

## Main Output

This stage prepares:

```text
Baseline C = pretrained expanded public dataset model
```

## Expected Split Sizes

| Split                        | Expected Count |
| ---------------------------- | -------------: |
| expanded known train         |           4869 |
| expanded known validation    |           1042 |
| expanded known test          |           1050 |
| expanded public unknown test |            318 |

## Important Research Decision

The RealWaste `Textile Trash` class remains separate as a public unknown/future-class split.

It is not forced into residual or any other current known class.

## Next Stage

The next stage should prepare the expanded pretrained training config and run a smoke test before full training.
