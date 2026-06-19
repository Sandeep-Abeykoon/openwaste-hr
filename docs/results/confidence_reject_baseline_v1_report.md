# Confidence-Threshold Reject Baseline v1

## Purpose

This report evaluates a simple reject-option baseline for OpenWaste-HR.

The trained closed-set classifier is allowed to reject low-confidence predictions instead of forcing every image into a known fine label.

## Classes

paper_cardboard, plastic, glass, metal, residual

## Selected Threshold

The threshold was selected using the validation split only.

| item | value |
| --- | --- |
| threshold | 0.64 |
| total_samples | 377.0 |
| accepted_count | 267.0 |
| rejected_count | 110.0 |
| coverage | 0.708223 |
| rejection_rate | 0.291777 |
| forced_accuracy | 0.718833 |
| selective_accuracy | 0.801498 |
| selective_error_rate | 0.198502 |
| selective_macro_f1 | 0.786836 |
| selective_weighted_f1 | 0.805563 |
| selection_metric | selective_macro_f1 |
| min_coverage | 0.7 |

## Validation Metrics After Rejection

| metric | value |
| --- | --- |
| total_samples | 377.0 |
| accepted_count | 267.0 |
| rejected_count | 110.0 |
| coverage | 0.708223 |
| rejection_rate | 0.291777 |
| forced_accuracy | 0.718833 |
| selective_accuracy | 0.801498 |
| selective_error_rate | 0.198502 |
| selective_macro_f1 | 0.786836 |
| selective_weighted_f1 | 0.805563 |

## Test Metrics After Rejection

| metric | value |
| --- | --- |
| total_samples | 384.0 |
| accepted_count | 262.0 |
| rejected_count | 122.0 |
| coverage | 0.682292 |
| rejection_rate | 0.317708 |
| forced_accuracy | 0.692708 |
| selective_accuracy | 0.770992 |
| selective_error_rate | 0.229008 |
| selective_macro_f1 | 0.715164 |
| selective_weighted_f1 | 0.774436 |

## Coverage-Risk Curve

![Coverage-risk curve](figures/confidence_reject_coverage_risk_v1.png)

## Test Confidence Histogram

![Confidence histogram](figures/confidence_reject_confidence_histogram_v1.png)

## Research Interpretation

This confidence-threshold baseline is the first safety-oriented baseline after the closed-set classifier.

It measures whether low-confidence predictions can be routed to manual review. This is important because the final OpenWaste-HR system should not only classify known items, but also reduce unsafe confident errors by rejecting uncertain, ambiguous, or unknown inputs.

This is still not the final proposed model. Later stages will add open-set scoring and hierarchical coarse/fine fallback.
