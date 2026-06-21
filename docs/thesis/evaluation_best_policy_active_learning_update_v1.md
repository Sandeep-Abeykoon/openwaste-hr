# Evaluation Update: Best Policy and Active Learning v2

## Purpose

This section updates the OpenWaste-HR evaluation chapter with the latest best model, best decision policy, live prototype result, and active learning v2 dataset planning result.

The evaluation now shows that the best current OpenWaste-HR system is not simply the model with the highest closed-set accuracy. The best system is the one that balances known-class performance, accepted-decision reliability, and local unknown safety.

## Best Current System

The best current system is:

```text
Pretrained Safe Hierarchical Policy
```

This system uses the pretrained transfer-learning model and the tuned safe hierarchical decision policy.

Selected thresholds:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

## Closed-Set Model Improvement

The pretrained model substantially improved closed-set known-class performance compared with the scratch-trained baseline.

| Model                    | Test Accuracy | Balanced Accuracy |  Macro-F1 | Weighted-F1 |
| ------------------------ | ------------: | ----------------: | --------: | ----------: |
| Scratch-trained baseline |      0.692708 |          0.654500 |  0.645600 |    0.700900 |
| Pretrained baseline      |      0.888000 |          0.843100 |  0.851000 |    0.887300 |
| Improvement              |     +0.195292 |         +0.188600 | +0.205400 |   +0.186400 |

This result shows that pretrained image features are valuable for the waste classification task.

## Best Policy Performance

The best policy result was obtained by tuning the pretrained hierarchical policy for safer open-set behaviour.

| Metric                                  |    Value |
| --------------------------------------- | -------: |
| Known-test coverage                     | 0.864583 |
| Known manual-review rate                | 0.135417 |
| Fine accuracy on fine decisions         | 0.958015 |
| Coarse accuracy on coarse decisions     | 0.971429 |
| Hierarchical success rate over accepted | 0.960843 |
| Local unknown manual-review rate        | 0.600000 |
| Local unknown acceptance rate           | 0.400000 |

This result is stronger than the earlier scratch safe hierarchical policy.

| Metric                           | Scratch Safe Hierarchical | Pretrained Safe Hierarchical |
| -------------------------------- | ------------------------: | ---------------------------: |
| Known coverage                   |                  0.658854 |                     0.864583 |
| Accepted reliability             |                  0.889328 |                     0.960843 |
| Local unknown manual-review rate |                  0.375000 |                     0.600000 |
| Local unknown acceptance rate    |                  0.625000 |                     0.400000 |

## Why the Safe Policy Was Needed

The first pretrained hierarchical policy had very high known-test coverage, but it was too permissive on local unknown images.

| Policy                              | Known Coverage | Accepted Reliability | Local Unknown Manual Review Rate | Local Unknown Acceptance Rate |
| ----------------------------------- | -------------: | -------------------: | -------------------------------: | ----------------------------: |
| Pretrained hierarchical policy v1   |       0.976562 |             0.957333 |                         0.075000 |                      0.925000 |
| Pretrained safe hierarchical policy |       0.864583 |             0.960843 |                         0.600000 |                      0.400000 |

This comparison shows that high known-test coverage alone is not enough. The safe policy reduces unsafe local unknown acceptance while keeping strong accepted-decision reliability.

## Live Prototype Evaluation

The best pretrained safe policy was integrated into the inference pipeline, API wrapper, backend endpoint, and frontend demo.

The live backend/frontend test used:

```text
ml/data/local_unknown/images/local_000001.jpg
```

This image was visually identified as a rubber slipper / flip-flop. It is outside the current known taxonomy.

The live prototype returned:

| Field                  | Value                  |
| ---------------------- | ---------------------- |
| Predicted fine label   | paper_cardboard        |
| Max softmax confidence | 0.993654               |
| Top coarse label       | recyclable             |
| Top coarse confidence  | 0.999999               |
| Decision type          | coarse_label           |
| Final label            | recyclable             |
| Final confidence       | 0.999999               |
| Decision reason        | coarse_fallback_stable |

This live result demonstrates an important open-world limitation: a model can produce high confidence even for a local object outside the known training taxonomy.

## Active Learning v2 Result

The reviewed local-label seed confirmed that:

| Field              | Value                          |
| ------------------ | ------------------------------ |
| Sample ID          | local_000001                   |
| Human observation  | rubber slipper / flip-flop     |
| Taxonomy status    | outside_current_known_taxonomy |
| Recommended action | keep_as_unknown_test           |
| Proposed new label | rubber_slipper_flip_flop       |

The active learning v2 dataset plan assigned the reviewed sample as:

| Role                              | Decision                                |
| --------------------------------- | --------------------------------------- |
| Include in known training v2      | false                                   |
| Include in unknown test v2        | true                                    |
| Include as future class candidate | true                                    |
| Active learning v2 role           | unknown_test_and_future_class_candidate |

This prevents the local unknown object from being incorrectly added to the known training set.

## Evaluation Interpretation

The evaluation supports the main OpenWaste-HR argument.

A normal closed-set classifier can produce good accuracy on known test data, but it does not know when an input is outside its label space. Even the pretrained model produced a high-confidence prediction for the rubber slipper image.

The final evaluation therefore shows that OpenWaste-HR needs more than a classifier. The strongest result comes from combining:

* pretrained image classification
* hierarchical fine/coarse taxonomy
* fine_label, coarse_label, and manual_review decisions
* local unknown evaluation
* safe threshold tuning
* backend/frontend prototype integration
* human-in-the-loop active learning

## Current Conclusion

The pretrained safe hierarchical policy is the best current OpenWaste-HR system.

It provides the strongest balance between known-test coverage, accepted-decision reliability, and local unknown safety. The active learning v2 workflow further strengthens the system by showing how reviewed local unknown objects can be kept for open-set testing or future taxonomy expansion instead of being forced into existing known classes.
