# Pretrained Training Smoke Test Summary v1

## Purpose

This stage checks whether the pretrained transfer-learning model setup can run before full training.

## Current Baseline

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |

## Smoke-Test Model

| Model                 | Description                              |
| --------------------- | ---------------------------------------- |
| Baseline B smoke test | pretrained transfer-learning setup check |

Smoke-test config:

```text id="391qtl"
ml/configs/train_pretrained_trashnet_smoke.yaml
```

Important setting:

```yaml id="8jvzu1"
pretrained: true
```

Separate output name:

```text id="oh2l7n"
pretrained_trashnet_smoke_v1
```

## What This Stage Confirms

| Check                            | Meaning                                               |
| -------------------------------- | ----------------------------------------------------- |
| pretrained config works          | the pretrained setting is accepted                    |
| model can start training         | pretrained training setup is usable                   |
| outputs are separated            | smoke-test files do not overwrite baseline outputs    |
| full training is safe to attempt | the next stage can run the full pretrained experiment |

## Not a Final Result

This smoke test is not used as the final pretrained model comparison.

The final comparison will be done after full pretrained training and evaluation.
