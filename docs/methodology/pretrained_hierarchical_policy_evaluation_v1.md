# Pretrained Hierarchical Policy Evaluation v1

## Purpose

This stage evaluates the hierarchical decision policy using the full pretrained transfer-learning checkpoint.

The scratch-trained model is treated as:

```text id="azm7j1"
Baseline A = scratch-trained TrashNet-style model
```

The pretrained model is treated as:

```text id="wk4hsr"
Baseline B = pretrained transfer-learning model
```

The purpose is to check whether the stronger pretrained classifier improves the OpenWaste-HR hierarchical decision workflow.

## Config

This stage uses:

```text id="054qyn"
ml/configs/pretrained_hierarchical_decision_policy.yaml
```

The pretrained checkpoint is:

```text id="nchztr"
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

## Decision Types

The hierarchical policy can return three decision types:

| Decision Type | Meaning                                |
| ------------- | -------------------------------------- |
| fine_label    | accept a detailed known fine label     |
| coarse_label  | return a safer broad fallback category |
| manual_review | route the image to human review        |

## Policy Setting

The pretrained model uses a stricter fine-label confidence threshold based on the pretrained confidence-reject evaluation.

| Threshold                     |  Value |
| ----------------------------- | -----: |
| fine_confidence_threshold     | 0.9900 |
| coarse_confidence_threshold   | 0.8000 |
| coarse_margin_threshold       | 0.1500 |
| minimum_confidence_for_coarse | 0.3500 |

The fine threshold is higher than the original scratch-trained hierarchical policy because the pretrained model produces stronger confidence values.

## What This Stage Measures

This evaluation measures the policy on:

| Dataset               | Purpose                                                                    |
| --------------------- | -------------------------------------------------------------------------- |
| known test set        | check fine/coarse/manual-review behaviour on known classes                 |
| local unknown dataset | check whether local unknown images are routed to manual review or accepted |

The main metrics are:

| Metric                                  | Meaning                                                     |
| --------------------------------------- | ----------------------------------------------------------- |
| fine_decision_count                     | number of fine-label decisions                              |
| coarse_fallback_count                   | number of coarse-label decisions                            |
| manual_review_count                     | number routed to manual review                              |
| known_decision_coverage                 | proportion of known samples accepted automatically          |
| hierarchical_success_rate_over_accepted | accepted-decision reliability                               |
| unknown_manual_review_rate              | proportion of local unknown samples routed to manual review |
| unknown_acceptance_rate                 | proportion of local unknown samples accepted                |

## Comparison Reference

The scratch-trained first hierarchical policy previously achieved:

| Metric                             |    Value |
| ---------------------------------- | -------: |
| Known decision coverage            | 0.932292 |
| Hierarchical success over accepted | 0.824022 |
| Local unknown manual-review rate   | 0.075000 |
| Local unknown acceptance rate      | 0.925000 |

The pretrained evaluation will show whether the stronger classifier improves hierarchical decision quality.

## Research Meaning

This stage is important because OpenWaste-HR is not only trying to improve normal accuracy. It is trying to improve decision safety.

If the pretrained hierarchical policy improves accepted-decision reliability and routes more local unknown samples to manual review, it supports using the pretrained checkpoint as the stronger base model for the final OpenWaste-HR workflow.

If it still accepts many local unknown samples, this confirms the need for safe hierarchical policy tuning in the next stage.
