# Open-Set Score Reject Baseline v1

## Purpose

This report evaluates two logit-based reject baselines:

1. Maximum Logit Score
2. Energy Score

The goal is to compare these against the previous softmax-confidence reject baseline.

## Classes

paper_cardboard, plastic, glass, metal, residual

## Selected Thresholds

Thresholds were selected using the validation split only.

| score_name | threshold | accept_direction | validation_coverage | validation_selective_accuracy | validation_selective_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| max_logit | 2.185518 | greater_equal | 0.732095 | 0.797101 | 0.777376 |
| energy | -2.614076 | less_equal | 0.732095 | 0.782609 | 0.75795 |

## Test Metrics

| score_name | metric | value |
| --- | --- | --- |
| max_logit | total_samples | 384.0 |
| max_logit | accepted_count | 275.0 |
| max_logit | rejected_count | 109.0 |
| max_logit | coverage | 0.716146 |
| max_logit | rejection_rate | 0.283854 |
| max_logit | forced_accuracy | 0.692708 |
| max_logit | selective_accuracy | 0.76 |
| max_logit | selective_error_rate | 0.24 |
| max_logit | selective_macro_f1 | 0.723257 |
| max_logit | selective_weighted_f1 | 0.763108 |
| energy | total_samples | 384.0 |
| energy | accepted_count | 286.0 |
| energy | rejected_count | 98.0 |
| energy | coverage | 0.744792 |
| energy | rejection_rate | 0.255208 |
| energy | forced_accuracy | 0.692708 |
| energy | selective_accuracy | 0.741259 |
| energy | selective_error_rate | 0.258741 |
| energy | selective_macro_f1 | 0.705525 |
| energy | selective_weighted_f1 | 0.745881 |

## Coverage-Risk Plot

![Open-set score coverage-risk plot](figures/open_set_score_coverage_risk_v1.png)

## Score Histogram Plot

![Open-set score histogram plot](figures/open_set_score_histogram_v1.png)

## Research Interpretation

This stage prepares OpenWaste-HR for true open-set and unknown-item evaluation.

Maximum logit and energy scoring use raw model logits instead of only softmax confidence. This is important because later unknown and local difficult images may expose overconfident softmax behaviour.

Current limitation: this report still evaluates selective classification on known TrashNet test images. The next dataset stage must introduce held-out unknown and local unknown images to test true unknown detection.
