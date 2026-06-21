# Final Model and Policy Comparison Summary v1

## Purpose

This document summarises the current best OpenWaste-HR model and policy comparison.

This is the first full comparison after pretrained training and safe pretrained hierarchical tuning.

## Main Result

The best current system is:

```text id="c6ycga"
Pretrained Safe Hierarchical Policy
```

## Closed-Set Improvement

| Model                    | Accuracy | Macro-F1 |
| ------------------------ | -------: | -------: |
| Scratch-trained baseline | 0.692708 | 0.645600 |
| Pretrained baseline      | 0.888000 | 0.851000 |

The pretrained model substantially improves known-class classification.

## Best Policy Result

| Metric                            |    Value |
| --------------------------------- | -------: |
| Known-test coverage               | 0.864583 |
| Accepted hierarchical reliability | 0.960843 |
| Local unknown manual-review rate  | 0.600000 |
| Local unknown acceptance rate     | 0.400000 |

## Why This Is Better Than the Previous Safe Policy

| Metric                           | Scratch Safe Policy | Pretrained Safe Policy |
| -------------------------------- | ------------------: | ---------------------: |
| Known coverage                   |            0.658854 |               0.864583 |
| Accepted reliability             |            0.889328 |               0.960843 |
| Local unknown manual-review rate |            0.375000 |               0.600000 |
| Local unknown acceptance rate    |            0.625000 |               0.400000 |

## Key Interpretation

The pretrained model improves known-test classification, but the safe hierarchical policy is still necessary.

The first pretrained hierarchical policy had very high known coverage, but accepted too many local unknown images. The tuned safe pretrained policy provides a better balance by preserving strong known reliability while routing more local unknown samples to manual review.

## Current Thesis Message

The strongest current thesis message is:

```text id="1c2ifi"
OpenWaste-HR improves waste classification by combining pretrained image recognition with hierarchical open-set decision-making, reject/manual-review routing, local unknown evaluation, and active learning support.
```

## Remaining Work

This comparison is not the final research endpoint.

Next work should include:

1. fill human correction labels
2. create reviewed local dataset v2
3. add more public datasets
4. retrain or fine-tune the model
5. compare the active-learning model with the current best policy
6. update the final thesis evaluation
