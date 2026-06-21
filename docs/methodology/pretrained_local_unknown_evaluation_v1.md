# Pretrained Local Unknown Evaluation v1

## Purpose

This stage evaluates the full pretrained transfer-learning checkpoint on the local unknown dataset.

The scratch-trained model is treated as:

```text
Baseline A = scratch-trained TrashNet-style model
```

The pretrained model is treated as:

```text
Baseline B = pretrained transfer-learning model
```

The purpose is to test whether the stronger pretrained classifier also improves unknown handling, or whether it becomes more confident on local images that are outside the known training distribution.

## Config

This stage uses:

```text
ml/configs/pretrained_local_unknown_evaluation.yaml
```

The pretrained checkpoint is:

```text
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

The evaluation uses the pretrained reject-option thresholds produced in the previous stage:

| Method               | Threshold Source                                        |
| -------------------- | ------------------------------------------------------- |
| confidence threshold | pretrained_confidence_reject_selected_threshold_v1.json |
| max-logit and energy | pretrained_open_set_score_selected_thresholds_v1.json   |

## Dataset

The local unknown dataset is used as an unknown-test set.

These images are not treated as normal known training classes. They are used to test whether the system can avoid unsafe acceptance of local, unfamiliar, or out-of-distribution waste images.

## Methods Evaluated

The same three reject methods are evaluated:

| Method               | Score Used                 |
| -------------------- | -------------------------- |
| confidence threshold | maximum softmax confidence |
| max-logit score      | highest raw logit          |
| energy score         | energy value from logits   |

## Metrics

The main metrics are:

| Metric                          | Meaning                                         |
| ------------------------------- | ----------------------------------------------- |
| total_unknown_samples           | number of local unknown images evaluated        |
| rejected_unknown_count          | number routed to rejection/manual review        |
| accepted_unknown_as_known_count | number incorrectly accepted as known            |
| unknown_rejection_rate          | proportion of unknown samples rejected          |
| unknown_false_acceptance_rate   | proportion of unknown samples accepted as known |

## Comparison with Scratch-Trained Local Unknown Evaluation

The scratch-trained local unknown result was:

| Method               | Unknown Rejection Rate | Unknown False Acceptance Rate |
| -------------------- | ---------------------: | ----------------------------: |
| Confidence threshold |               0.350000 |                      0.650000 |
| Max-logit score      |               0.275000 |                      0.725000 |
| Energy score         |               0.200000 |                      0.800000 |

This stage will compare the pretrained model against these results.

## Research Meaning

This is an important evaluation stage because stronger known-class accuracy does not automatically guarantee better unknown handling.

If the pretrained model rejects more local unknown images, it improves both recognition and safety. If it accepts more unknown images, this shows that improved closed-set accuracy must still be combined with careful open-set and hierarchical decision policies.
