# Expanded Public Pretrained Training Summary v1

## Purpose

This stage runs the full expanded public pretrained training experiment.

## Baseline C

```text id="f2tg5l"
Baseline C = pretrained expanded public dataset model
```

## Training Data

| Split      | Count |
| ---------- | ----: |
| train      |  4869 |
| validation |  1042 |
| test       |  1050 |

## Known Classes

The model trains on:

* paper_cardboard
* plastic
* glass
* metal
* organic
* residual

The `unknown` class is excluded from training.

## Important Research Point

This experiment checks whether adding RealWaste to the original TrashNet-style dataset improves the model compared with the previous TrashNet-only pretrained baseline.

## Next Stage

After training, the model should be evaluated using the same closed-set evaluation process so it can be compared fairly against the earlier models.
