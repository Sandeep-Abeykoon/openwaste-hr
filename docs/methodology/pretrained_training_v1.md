# Full Pretrained Model Training v1

## Purpose

This stage trains the first full pretrained transfer-learning model for OpenWaste-HR.

The current scratch-trained model is treated as:

```text
Baseline A = scratch-trained TrashNet-style model
```

The model trained in this stage is:

```text
Baseline B = pretrained transfer-learning model
```

The purpose is to test whether pretrained visual features improve known-class waste classification and later improve the OpenWaste-HR hierarchical decision workflow.

## Training Config

The full pretrained training uses:

```text
ml/configs/train_pretrained_trashnet.yaml
```

The important setting is:

```yaml
pretrained: true
```

The expected output naming is:

```text
pretrained_trashnet_v1
```

This keeps the pretrained training outputs separate from the original scratch-trained baseline outputs.

## Why Pretrained Training Is Used

The original scratch-trained baseline was useful as a first controlled comparison point. However, the available known dataset is relatively small and visually varied.

A pretrained model starts with visual features learned from a larger image dataset and then adapts them to waste classification. This may improve recognition of shapes, textures, colours, packaging materials, and object boundaries.

## Evaluation After Training

After training, the pretrained model will be evaluated using:

| Evaluation Area                | Purpose                                                  |
| ------------------------------ | -------------------------------------------------------- |
| closed-set metrics             | compare known-test performance with Baseline A           |
| confidence reject baseline     | test selective prediction reliability                    |
| max-logit and energy baselines | compare open-set scoring methods                         |
| local unknown evaluation       | test unknown rejection and false acceptance              |
| hierarchical policy evaluation | test fine_label, coarse_label, and manual_review outputs |
| safe policy tuning             | compare safety-coverage trade-off                        |

## Comparison Plan

The first comparison will be:

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |
| Baseline B | pretrained transfer-learning model   |

The comparison will include:

| Metric                               | Purpose                              |
| ------------------------------------ | ------------------------------------ |
| test accuracy                        | known-test correctness               |
| balanced accuracy                    | class-balanced correctness           |
| macro-F1                             | equal-weighted class performance     |
| weighted-F1                          | class-frequency-weighted performance |
| selective accuracy                   | accepted prediction reliability      |
| unknown rejection/manual-review rate | unknown safety                       |
| unknown false acceptance rate        | unsafe unknown acceptance            |

## Research Meaning

This stage begins the model-strength improvement phase of OpenWaste-HR.

The main contribution remains the hierarchical open-set decision workflow. The pretrained model tests whether stronger image features improve the quality of the classifier and therefore improve downstream reject/manual-review and hierarchical decision behaviour.
