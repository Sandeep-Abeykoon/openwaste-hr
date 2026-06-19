# Final Policy Comparison v1

## Purpose

This document compares the main OpenWaste-HR decision stages:

1. closed-set baseline classifier
2. confidence-threshold reject baseline
3. hierarchical decision policy v1
4. safe hierarchical policy tuning v1

The purpose is to show the project progression from forced classification toward safer open-world decision-making.

## Summary of Compared Methods

| Method                             | Output Types                               | Main Purpose                       |
| ---------------------------------- | ------------------------------------------ | ---------------------------------- |
| Closed-set baseline                | fine label only                            | first baseline classifier          |
| Confidence-threshold reject        | fine label or manual review                | simple reject-option baseline      |
| Hierarchical decision policy v1    | fine label, coarse label, or manual review | first hierarchical fallback policy |
| Safe hierarchical policy tuning v1 | fine label, coarse label, or manual review | safer tuned hierarchical policy    |

## Known-Test Comparison

| Method                             | Known Samples | Fine Decisions | Coarse Decisions | Manual Review / Rejected | Known Coverage |               Accepted-Decision Reliability |
| ---------------------------------- | ------------: | -------------: | ---------------: | -----------------------: | -------------: | ------------------------------------------: |
| Closed-set baseline                |           384 |            384 |                0 |                        0 |       1.000000 |                           0.692708 accuracy |
| Confidence-threshold reject        |           384 |            262 |                0 |                      122 |       0.682292 |                 0.770992 selective accuracy |
| Hierarchical decision policy v1    |           384 |            262 |               96 |                       26 |       0.932292 | 0.824022 hierarchical success over accepted |
| Safe hierarchical policy tuning v1 |           384 |            145 |              108 |                      131 |       0.658854 | 0.889328 hierarchical success over accepted |

## Local Unknown Dataset Comparison

| Method                             | Unknown Samples | Manual Review / Rejected | Fine Accepts | Coarse Accepts | Unknown Accepted | Unknown Manual Review Rate | Unknown Acceptance Rate |
| ---------------------------------- | --------------: | -----------------------: | -----------: | -------------: | ---------------: | -------------------------: | ----------------------: |
| Confidence-threshold reject        |              40 |                       14 |           26 |              0 |               26 |                   0.350000 |                0.650000 |
| Maximum Logit Score reject         |              40 |                       11 |           29 |              0 |               29 |                   0.275000 |                0.725000 |
| Energy Score reject                |              40 |                        8 |           32 |              0 |               32 |                   0.200000 |                0.800000 |
| Hierarchical decision policy v1    |              40 |                        3 |           26 |             11 |               37 |                   0.075000 |                0.925000 |
| Safe hierarchical policy tuning v1 |              40 |                       15 |           16 |              9 |               25 |                   0.375000 |                0.625000 |

## Main Findings

### Finding 1: Closed-set classification is not enough

The closed-set baseline always predicts a known fine label. This makes it unsuitable for unknown, ambiguous, or locally unusual waste images because it cannot express uncertainty.

### Finding 2: Simple rejection improves safety but loses coverage

The confidence-threshold reject baseline improved accepted known-test accuracy and rejected 14 out of 40 local unknown samples. However, it still accepted 26 out of 40 local unknown samples as known labels.

### Finding 3: Hierarchical fallback improves known-test usefulness

Hierarchical decision policy v1 produced high known-test coverage because it allowed uncertain fine predictions to fall back to coarse labels. It accepted 358 out of 384 known-test samples through either fine or coarse output.

However, this policy was too permissive for local unknown images because only 3 out of 40 local unknown samples were sent to manual review.

### Finding 4: Safe tuning improves local unknown handling

Safe hierarchical policy tuning selected stricter thresholds:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.900000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.650000 |
| minimum_confidence_for_coarse | 0.650000 |

The tuned policy increased the local unknown manual-review rate from 0.075000 to 0.375000.

This means manual-review routing improved from 3 out of 40 local unknown samples to 15 out of 40 local unknown samples.

### Finding 5: There is a coverage-safety trade-off

The tuned policy improves safety but reduces known-test coverage.

| Policy                             | Known Coverage | Local Unknown Manual Review Rate |
| ---------------------------------- | -------------: | -------------------------------: |
| Hierarchical decision policy v1    |       0.932292 |                         0.075000 |
| Safe hierarchical policy tuning v1 |       0.658854 |                         0.375000 |

This trade-off is important for the thesis because it shows that open-world waste classification should not be judged only by accuracy. It must also consider uncertainty, rejection, manual review, and safety.

## Final Current Decision

The selected current OpenWaste-HR decision policy is:

**Safe hierarchical policy tuning v1**

Reason:

* it supports fine-label, coarse-label, and manual-review outputs
* it improves local unknown manual-review routing
* it has the highest accepted-decision reliability among the compared hierarchical policies
* it better matches the OpenWaste-HR goal of avoiding unsafe forced predictions

## Limitation

The tuned policy is still not perfect. It accepts 25 out of 40 local unknown samples as either fine or coarse outputs.

This means future work should improve open-set detection using stronger feature representations, better calibrated confidence, additional local data, or active learning.

## Thesis-Ready Conclusion

The experiment shows that moving from closed-set classification to hierarchical reject-aware decision-making improves the system’s ability to manage uncertainty. The safest current policy is the tuned hierarchical policy, which increases manual-review routing for local unknown images while maintaining reliable accepted decisions on known-test samples.
