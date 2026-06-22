# Expanded Public Unknown Evaluation Summary v1

## Purpose

This stage evaluates the expanded public pretrained model against unknown/outside-taxonomy samples.

## Baseline C

```text id="0evzwj"
Baseline C = pretrained expanded public dataset model
```

## Unknown Sources

| Source                            | Manifest                                           |
| --------------------------------- | -------------------------------------------------- |
| local unknown dataset             | ml/data/splits/local_unknown_manifest_v1.csv       |
| public unknown/future-class split | ml/data/splits/expanded_public_unknown_test_v1.csv |

## Public Unknown Split

The public unknown split comes from RealWaste `Textile Trash`.

This class remains outside the current known taxonomy and is used as unknown/future-class evaluation data.

## Research Point

This stage checks whether dataset expansion improves unknown-handling behaviour, not only closed-set accuracy.

## Next Stage

After this evaluation, the next stage should tune the expanded public safe hierarchical policy.
