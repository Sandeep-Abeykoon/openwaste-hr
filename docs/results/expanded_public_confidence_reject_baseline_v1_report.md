# Confidence-Threshold Reject Baseline v1

## Purpose

This report evaluates a simple reject-option baseline for OpenWaste-HR.

The trained closed-set classifier is allowed to reject low-confidence predictions instead of forcing every image into a known fine label.

## Classes

paper_cardboard, plastic, glass, metal, organic, residual

## Selected Threshold

The threshold was selected using the validation split only.

| item | value |
| --- | --- |
| threshold | 0.99 |
| total_samples | 1042.0 |
| accepted_count | 756.0 |
| rejected_count | 286.0 |
| coverage | 0.725528 |
| rejection_rate | 0.274472 |
| forced_accuracy | 0.894434 |
| selective_accuracy | 0.977513 |
| selective_error_rate | 0.022487 |
| selective_macro_f1 | 0.972882 |
| selective_weighted_f1 | 0.977556 |
| selection_metric | selective_macro_f1 |
| min_coverage | 0.7 |

## Validation Metrics After Rejection

| metric | value |
| --- | --- |
| total_samples | 1042.0 |
| accepted_count | 756.0 |
| rejected_count | 286.0 |
| coverage | 0.725528 |
| rejection_rate | 0.274472 |
| forced_accuracy | 0.894434 |
| selective_accuracy | 0.977513 |
| selective_error_rate | 0.022487 |
| selective_macro_f1 | 0.972882 |
| selective_weighted_f1 | 0.977556 |

## Test Metrics After Rejection

| metric | value |
| --- | --- |
| total_samples | 1050.0 |
| accepted_count | 759.0 |
| rejected_count | 291.0 |
| coverage | 0.722857 |
| rejection_rate | 0.277143 |
| forced_accuracy | 0.887619 |
| selective_accuracy | 0.97892 |
| selective_error_rate | 0.02108 |
| selective_macro_f1 | 0.973239 |
| selective_weighted_f1 | 0.978761 |

## Coverage-Risk Curve

![Coverage-risk curve](figures/expanded_public_confidence_reject_coverage_risk_v1.png)

## Test Confidence Histogram

![Confidence histogram](figures/expanded_public_confidence_reject_confidence_histogram_v1.png)

## Research Interpretation

This confidence-threshold baseline is the first safety-oriented baseline after the closed-set classifier.

It measures whether low-confidence predictions can be routed to manual review. This is important because the final OpenWaste-HR system should not only classify known items, but also reduce unsafe confident errors by rejecting uncertain, ambiguous, or unknown inputs.

This is still not the final proposed model. Later stages will add open-set scoring and hierarchical coarse/fine fallback.
