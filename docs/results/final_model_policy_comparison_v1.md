# Final Model and Policy Comparison v1

## Purpose

This document compares the main OpenWaste-HR model and decision-policy results produced so far.

This is the first complete comparison after training the pretrained transfer-learning model and tuning the safe pretrained hierarchical policy.

This comparison is still a v1 comparison. It will be updated later after:

* filling human correction labels
* creating reviewed local dataset v2
* adding more public waste datasets
* retraining or fine-tuning with active-learning feedback

## Models Compared

| Model      | Description                          |
| ---------- | ------------------------------------ |
| Baseline A | scratch-trained TrashNet-style model |
| Baseline B | pretrained transfer-learning model   |

## Decision Policies Compared

| Policy                              | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------------- |
| Closed-set baseline                 | always forces a known fine label                                    |
| Confidence reject                   | accepts confident fine-label predictions, rejects uncertain samples |
| Hierarchical policy v1              | returns fine_label, coarse_label, or manual_review                  |
| Safe hierarchical policy            | stricter tuned hierarchical policy for safer unknown handling       |
| Pretrained safe hierarchical policy | tuned safe policy using the pretrained model                        |

## Closed-Set Model Comparison

| Model                       | Test Accuracy | Balanced Accuracy |  Macro-F1 | Weighted-F1 |
| --------------------------- | ------------: | ----------------: | --------: | ----------: |
| Baseline A: scratch-trained |      0.692708 |          0.654500 |  0.645600 |    0.700900 |
| Baseline B: pretrained      |      0.888000 |          0.843100 |  0.851000 |    0.887300 |
| Improvement                 |     +0.195292 |         +0.188600 | +0.205400 |   +0.186400 |

## Known-Test Decision Policy Comparison

| System                              | Known Coverage | Accepted Reliability | Notes                                   |
| ----------------------------------- | -------------: | -------------------: | --------------------------------------- |
| Closed-set scratch baseline         |       1.000000 |             0.692708 | forced fine-label accuracy              |
| Scratch confidence reject           |       0.682292 |             0.770992 | selective accuracy                      |
| Scratch hierarchical policy v1      |       0.932292 |             0.824022 | hierarchical success over accepted      |
| Scratch safe hierarchical policy    |       0.658854 |             0.889328 | safer but lower coverage                |
| Closed-set pretrained baseline      |       1.000000 |             0.888000 | forced fine-label accuracy              |
| Pretrained confidence reject        |       0.721354 |             0.953069 | derived selective accuracy              |
| Pretrained max-logit reject         |       0.734375 |             0.957447 | selective accuracy                      |
| Pretrained energy reject            |       0.744792 |             0.958042 | selective accuracy                      |
| Pretrained hierarchical policy v1   |       0.976562 |             0.957333 | very high coverage, weak unknown safety |
| Pretrained safe hierarchical policy |       0.864583 |             0.960843 | best current safety-coverage balance    |

## Local Unknown Handling Comparison

| System                              | Local Unknown Manual Review / Rejection Rate | Local Unknown Acceptance Rate |
| ----------------------------------- | -------------------------------------------: | ----------------------------: |
| Closed-set scratch baseline         |                                     0.000000 |                      1.000000 |
| Scratch confidence reject           |                                     0.350000 |                      0.650000 |
| Scratch max-logit reject            |                                     0.275000 |                      0.725000 |
| Scratch energy reject               |                                     0.200000 |                      0.800000 |
| Scratch hierarchical policy v1      |                                     0.075000 |                      0.925000 |
| Scratch safe hierarchical policy    |                                     0.375000 |                      0.625000 |
| Closed-set pretrained baseline      |                                     0.000000 |                      1.000000 |
| Pretrained confidence reject        |                                     0.500000 |                      0.500000 |
| Pretrained max-logit reject         |                                     0.250000 |                      0.750000 |
| Pretrained energy reject            |                                     0.250000 |                      0.750000 |
| Pretrained hierarchical policy v1   |                                     0.075000 |                      0.925000 |
| Pretrained safe hierarchical policy |                                     0.600000 |                      0.400000 |

## Best Current Policy

The best current OpenWaste-HR policy is:

```text id="76m5nq"
Pretrained Safe Hierarchical Policy
```

Selected thresholds:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

Known-test result:

| Metric                                  |    Value |
| --------------------------------------- | -------: |
| Known total samples                     |      384 |
| Fine decisions                          |      262 |
| Coarse fallback decisions               |       70 |
| Manual-review decisions                 |       52 |
| Known decision coverage                 | 0.864583 |
| Known manual-review rate                | 0.135417 |
| Hierarchical success rate over accepted | 0.960843 |

Local unknown result:

| Metric                      |    Value |
| --------------------------- | -------: |
| Unknown total samples       |       40 |
| Unknown manual-review count |       24 |
| Unknown fine accept count   |        6 |
| Unknown coarse accept count |       10 |
| Unknown accepted count      |       16 |
| Unknown manual-review rate  | 0.600000 |
| Unknown acceptance rate     | 0.400000 |

## Why This Policy Is Best So Far

The pretrained safe hierarchical policy is selected as the best current OpenWaste-HR policy because it improves all three important areas:

| Area                          | Improvement                                                                |
| ----------------------------- | -------------------------------------------------------------------------- |
| Known classification          | pretrained model improves known-test accuracy from 0.692708 to 0.888000    |
| Accepted decision reliability | safe pretrained policy achieves 0.960843 accepted hierarchical reliability |
| Local unknown safety          | local unknown manual-review rate improves to 0.600000                      |

Compared with the scratch safe hierarchical policy:

| Metric                           | Scratch Safe Hierarchical | Pretrained Safe Hierarchical |
| -------------------------------- | ------------------------: | ---------------------------: |
| Known coverage                   |                  0.658854 |                     0.864583 |
| Accepted reliability             |                  0.889328 |                     0.960843 |
| Local unknown manual-review rate |                  0.375000 |                     0.600000 |
| Local unknown acceptance rate    |                  0.625000 |                     0.400000 |

## Main Research Interpretation

The results show that using a pretrained model improves known-class performance substantially. However, the local unknown results also show that a stronger classifier alone does not fully solve open-set waste classification.

For example, the pretrained hierarchical policy v1 achieved very strong known-test coverage of 0.976562, but only routed 0.075000 of local unknown images to manual review. This means the model was strong on known classes but still too permissive for unknown images.

The safe pretrained hierarchical policy solved this better by increasing local unknown manual-review rate to 0.600000 while keeping useful known-test coverage of 0.864583 and high accepted-decision reliability of 0.960843.

Therefore, the main OpenWaste-HR contribution is not only the pretrained classifier. The stronger contribution is the combination of:

* pretrained image classification
* hierarchical fine/coarse taxonomy
* fine_label, coarse_label, and manual_review decisions
* local unknown evaluation
* safe threshold tuning
* active learning support

## Current Best System Summary

| Item                              | Current Best Choice                            |
| --------------------------------- | ---------------------------------------------- |
| Model                             | Baseline B: pretrained transfer-learning model |
| Decision policy                   | pretrained safe hierarchical policy            |
| Known-test coverage               | 0.864583                                       |
| Accepted hierarchical reliability | 0.960843                                       |
| Local unknown manual-review rate  | 0.600000                                       |
| Local unknown acceptance rate     | 0.400000                                       |

## Remaining Work

This is not the final research comparison yet.

The next major improvements are:

1. fill human correction labels
2. process reviewed local labels
3. create reviewed local dataset v2
4. add additional public waste datasets
5. retrain or fine-tune using active-learning feedback
6. compare the updated model against the current pretrained safe policy
7. update the thesis evaluation chapter with final comparison results

## Conclusion

The current comparison shows that the pretrained safe hierarchical policy is the best OpenWaste-HR system so far.

It provides the strongest balance between known-test performance, accepted-decision reliability, and local unknown safety. This result supports the project direction of hierarchical open-set waste classification with reject/manual-review decisions and local active learning.
