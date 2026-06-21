# Pretrained Training Plan v1

## Purpose

This document prepares the next model-training stage for OpenWaste-HR.

The current model is the first baseline model trained mainly on the TrashNet-style dataset. It was trained as the original comparison point.

The next model will be a pretrained transfer-learning model. This means the model will start from visual features learned from a larger image dataset, then adapt those features to the OpenWaste-HR waste classification task.

## Current Baseline

The current baseline is:

```text id="26n2n3"
Baseline A = scratch-trained TrashNet-style model
```

This baseline is useful because it provides the first comparison point for the project.

Current baseline result:

| Metric            |    Value |
| ----------------- | -------: |
| Test accuracy     | 0.692708 |
| Balanced accuracy | 0.654500 |
| Macro-F1          | 0.645600 |
| Weighted-F1       | 0.700900 |

## Planned Pretrained Model

The next model is:

```text id="k6widc"
Baseline B = pretrained transfer-learning model
```

The planned config file is:

```text id="xn7uwy"
ml/configs/train_pretrained_trashnet.yaml
```

The key setting is:

```yaml id="1cw4wp"
pretrained: true
```

The output directory should be separate from the scratch-trained baseline so the results are not mixed.

Expected output naming:

```text id="0qg7io"
pretrained_trashnet_v1
```

## Why Pretrained Training Is Needed

The scratch-trained model is useful as a starting point, but the dataset is relatively small. A pretrained model may perform better because it starts with more general visual features such as edges, textures, shapes, colours, and object patterns.

This is useful for waste classification because waste images can vary heavily in:

* lighting
* background
* object angle
* damage or dirt
* packaging design
* object shape
* local context

## Planned Comparison

After pretrained training, the result will be compared against the original baseline.

| Model      | Training Style               | Purpose                                              |
| ---------- | ---------------------------- | ---------------------------------------------------- |
| Baseline A | scratch-trained              | original comparison point                            |
| Baseline B | pretrained transfer learning | test whether pretrained features improve performance |

The comparison will include:

| Metric                           | Purpose                              |
| -------------------------------- | ------------------------------------ |
| test accuracy                    | overall known-test correctness       |
| balanced accuracy                | class-balanced correctness           |
| macro-F1                         | equal-weighted class performance     |
| weighted-F1                      | frequency-weighted class performance |
| selective accuracy               | accepted prediction reliability      |
| local unknown manual-review rate | unknown safety behaviour             |
| unknown false acceptance rate    | unsafe unknown acceptance            |

## Expected Workflow

The pretrained training workflow will be:

```text id="m0gc77"
prepare pretrained config → run smoke test → train pretrained model → evaluate known test set → evaluate reject options → evaluate local unknowns → apply safe hierarchical policy → compare with Baseline A
```

## Important Research Point

The pretrained model is not replacing the OpenWaste-HR contribution.

The main contribution remains:

```text id="0ia588"
hierarchical open-set waste classification with reject/manual-review decisions and local active learning
```

The pretrained model is used to test whether a stronger backbone improves the performance of this decision workflow.
