# Open-Set Score Reject Baseline v1

## Purpose

This report evaluates two logit-based reject baselines:

1. Maximum Logit Score
2. Energy Score

The goal is to compare these against the previous softmax-confidence reject baseline.

## Classes

paper_cardboard, plastic, glass, metal, organic, residual

## Selected Thresholds

Thresholds were selected using the validation split only.

| score_name | threshold | accept_direction | validation_coverage | validation_selective_accuracy | validation_selective_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| max_logit | 9.59305 | greater_equal | 0.71881 | 0.970628 | 0.968614 |
| energy | -9.870391 | less_equal | 0.701536 | 0.974008 | 0.971993 |

## Test Metrics

| score_name | metric | value |
| --- | --- | --- |
| max_logit | total_samples | 1050.0 |
| max_logit | accepted_count | 773.0 |
| max_logit | rejected_count | 277.0 |
| max_logit | coverage | 0.73619 |
| max_logit | rejection_rate | 0.26381 |
| max_logit | forced_accuracy | 0.887619 |
| max_logit | selective_accuracy | 0.967658 |
| max_logit | selective_error_rate | 0.032342 |
| max_logit | selective_macro_f1 | 0.962688 |
| max_logit | selective_weighted_f1 | 0.967578 |
| energy | total_samples | 1050.0 |
| energy | accepted_count | 754.0 |
| energy | rejected_count | 296.0 |
| energy | coverage | 0.718095 |
| energy | rejection_rate | 0.281905 |
| energy | forced_accuracy | 0.887619 |
| energy | selective_accuracy | 0.966844 |
| energy | selective_error_rate | 0.033156 |
| energy | selective_macro_f1 | 0.961199 |
| energy | selective_weighted_f1 | 0.96676 |

## Coverage-Risk Plot

![Open-set score coverage-risk plot](figures/expanded_public_open_set_score_coverage_risk_v1.png)

## Score Histogram Plot

![Open-set score histogram plot](figures/expanded_public_open_set_score_histogram_v1.png)

## Research Interpretation

This stage prepares OpenWaste-HR for true open-set and unknown-item evaluation.

Maximum logit and energy scoring use raw model logits instead of only softmax confidence. This is important because later unknown and local difficult images may expose overconfident softmax behaviour.

Current limitation: this report still evaluates selective classification on known TrashNet test images. The next dataset stage must introduce held-out unknown and local unknown images to test true unknown detection.
