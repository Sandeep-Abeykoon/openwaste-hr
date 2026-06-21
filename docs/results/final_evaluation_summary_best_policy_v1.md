# Final Evaluation Summary: Best Policy v1

## Best Current System

```text
Pretrained Safe Hierarchical Policy
```

## Main Result

| Metric                            |    Value |
| --------------------------------- | -------: |
| Known-test coverage               | 0.864583 |
| Accepted hierarchical reliability | 0.960843 |
| Local unknown manual-review rate  | 0.600000 |
| Local unknown acceptance rate     | 0.400000 |

## Closed-Set Model Improvement

| Model                    | Accuracy | Macro-F1 |
| ------------------------ | -------: | -------: |
| Scratch-trained baseline | 0.692708 | 0.645600 |
| Pretrained baseline      | 0.888000 | 0.851000 |

The pretrained model improves known-class accuracy and macro-F1 substantially.

## Best Policy Thresholds

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

## Best Policy Comparison

| Policy                       | Known Coverage | Accepted Reliability | Local Unknown Manual Review | Local Unknown Acceptance |
| ---------------------------- | -------------: | -------------------: | --------------------------: | -----------------------: |
| Scratch safe hierarchical    |       0.658854 |             0.889328 |                    0.375000 |                 0.625000 |
| Pretrained hierarchical v1   |       0.976562 |             0.957333 |                    0.075000 |                 0.925000 |
| Pretrained safe hierarchical |       0.864583 |             0.960843 |                    0.600000 |                 0.400000 |

## Live Demo Example

The live backend/frontend prototype was tested using:

```text
ml/data/local_unknown/images/local_000001.jpg
```

Human observation:

```text
rubber slipper / flip-flop
```

Model/API result:

| Field                  | Value           |
| ---------------------- | --------------- |
| Predicted fine label   | paper_cardboard |
| Max softmax confidence | 0.993654        |
| Decision type          | coarse_label    |
| Final label            | recyclable      |
| Final confidence       | 0.999999        |

## Active Learning v2 Decision

The reviewed local sample was assigned as:

| Field                             | Value                                   |
| --------------------------------- | --------------------------------------- |
| Taxonomy status                   | outside_current_known_taxonomy          |
| Recommended action                | keep_as_unknown_test                    |
| Include in known training v2      | false                                   |
| Include in unknown test v2        | true                                    |
| Include as future class candidate | true                                    |
| Active learning v2 role           | unknown_test_and_future_class_candidate |

## Final Evaluation Message

The final v1 evaluation shows that OpenWaste-HR should be presented as a hierarchical open-set waste classification workflow, not only as a closed-set image classifier.

The key result is that the pretrained safe hierarchical policy gives the best balance between known-test reliability and local unknown safety, while active learning v2 provides a pathway for human-reviewed local data to improve future versions of the system.
