# Full Pretrained Model Training Summary v1

## Purpose

This stage trains Baseline B, the pretrained transfer-learning model.

## Model Comparison Context

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |
| Baseline B | pretrained transfer-learning model   |

## Training Config

```text
ml/configs/train_pretrained_trashnet.yaml
```

Important setting:

```yaml
pretrained: true
```

Expected output name:

```text
pretrained_trashnet_v1
```

## Why This Stage Matters

The current working prototype uses the first scratch-trained baseline. The pretrained model may improve known-class classification because it starts from stronger image features.

After this model is trained, it will be evaluated and compared against the scratch-trained baseline.

## Not the Final System Alone

The pretrained model is not the whole project contribution.

The main project contribution remains:

```text
hierarchical open-set waste classification with reject/manual-review decisions and local active learning
```

The pretrained model is used to strengthen the classifier inside that decision workflow.
