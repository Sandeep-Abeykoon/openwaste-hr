# Expanded Public Pretrained Training Preparation Summary v1

## Purpose

This stage prepares Baseline C training.

## Baseline C

```text
Baseline C = pretrained expanded public dataset model
```

## Input Data

| Dataset                | Role                           |
| ---------------------- | ------------------------------ |
| TrashNet-style dataset | original baseline source       |
| RealWaste              | expanded public dataset source |

## Expanded Known Splits

| Split      | Count |
| ---------- | ----: |
| train      |  4869 |
| validation |  1042 |
| test       |  1050 |

## Configs Created

| Config                                                 | Purpose                                  |
| ------------------------------------------------------ | ---------------------------------------- |
| ml/configs/train_expanded_public_pretrained.yaml       | full expanded public pretrained training |
| ml/configs/train_expanded_public_pretrained_smoke.yaml | 1-epoch smoke test                       |

## Next Step

Run the smoke training first.

If the smoke test works, the next stage should run full expanded public pretrained training and compare it against the current TrashNet pretrained model.
