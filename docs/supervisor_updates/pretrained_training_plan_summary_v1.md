# Pretrained Training Plan Summary v1

## Purpose

The next model-development stage is to train a pretrained transfer-learning model and compare it with the current scratch-trained baseline.

## Current Model

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |

Current known-test result:

| Metric        |    Value |
| ------------- | -------: |
| Test accuracy | 0.692708 |
| Macro-F1      | 0.645600 |

## Planned Model

| Model      | Description                        |
| ---------- | ---------------------------------- |
| Baseline B | pretrained transfer-learning model |

Prepared config:

```text id="zw7w8d"
ml/configs/train_pretrained_trashnet.yaml
```

Important setting:

```yaml id="jj0xc7"
pretrained: true
```

## Why This Is Needed

The current model is a useful baseline, but it was trained as the first comparison point. A pretrained model may improve feature extraction and known-class classification performance.

This will allow the project to compare:

```text id="kmqgiu"
scratch-trained baseline vs pretrained transfer-learning baseline
```

## Planned Evaluation

The pretrained model will be compared using:

* test accuracy
* balanced accuracy
* macro-F1
* weighted-F1
* reject-option results
* local unknown rejection/manual-review rate
* unknown false acceptance rate
* safe hierarchical policy performance

## Research Position

The pretrained model is a model-strength improvement.

The main project contribution is still the OpenWaste-HR decision workflow:

```text id="izcl6d"
fine_label → coarse_label → manual_review
```

with local unknown evaluation and active learning support.
