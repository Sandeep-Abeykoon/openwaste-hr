# Safe Hierarchical Policy Tuning v1 Report

## Purpose

This report tunes the OpenWaste-HR hierarchical decision policy to improve local unknown safety while preserving useful known-test decisions.

## Selected Thresholds

| threshold | value |
| --- | --- |
| fine_confidence_threshold | 0.9 |
| coarse_confidence_threshold | 0.8 |
| coarse_margin_threshold | 0.65 |
| minimum_confidence_for_coarse | 0.65 |

## Known-Test Metrics

| metric | value |
| --- | --- |
| known_total_samples | 384.0 |
| fine_decision_count | 145.0 |
| coarse_fallback_count | 108.0 |
| manual_review_count | 131.0 |
| known_decision_coverage | 0.658854 |
| known_manual_review_rate | 0.341146 |
| hierarchical_success_rate_over_all | 0.585938 |
| hierarchical_success_rate_over_accepted | 0.889328 |

## Local Unknown Metrics

| metric | value |
| --- | --- |
| unknown_total_samples | 40.0 |
| unknown_manual_review_count | 15.0 |
| unknown_fine_accept_count | 16.0 |
| unknown_coarse_accept_count | 9.0 |
| unknown_accepted_count | 25.0 |
| unknown_manual_review_rate | 0.375 |
| unknown_acceptance_rate | 0.625 |

## Decision Distribution

| dataset | decision_type | count | percentage |
| --- | --- | --- | --- |
| known_test | fine_label | 145 | 37.76 |
| known_test | coarse_label | 108 | 28.12 |
| known_test | manual_review | 131 | 34.11 |
| local_unknown | fine_label | 16 | 40.0 |
| local_unknown | coarse_label | 9 | 22.5 |
| local_unknown | manual_review | 15 | 37.5 |

## Top Candidate Policies

| fine_confidence_threshold | coarse_confidence_threshold | coarse_margin_threshold | minimum_confidence_for_coarse | objective_score | known_decision_coverage | hierarchical_success_rate_over_accepted | unknown_manual_review_rate | unknown_acceptance_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9 | 0.8 | 0.65 | 0.65 | 0.586069 | 0.658854 | 0.889328 | 0.375 | 0.625 |
| 0.9 | 0.8 | 0.15 | 0.65 | 0.58554 | 0.661458 | 0.885827 | 0.375 | 0.625 |
| 0.9 | 0.8 | 0.25 | 0.65 | 0.58554 | 0.661458 | 0.885827 | 0.375 | 0.625 |
| 0.9 | 0.8 | 0.35 | 0.65 | 0.58554 | 0.661458 | 0.885827 | 0.375 | 0.625 |
| 0.9 | 0.8 | 0.5 | 0.65 | 0.58554 | 0.661458 | 0.885827 | 0.375 | 0.625 |
| 0.9 | 0.85 | 0.15 | 0.65 | 0.585308 | 0.651042 | 0.892 | 0.375 | 0.625 |
| 0.9 | 0.85 | 0.25 | 0.65 | 0.585308 | 0.651042 | 0.892 | 0.375 | 0.625 |
| 0.9 | 0.85 | 0.35 | 0.65 | 0.585308 | 0.651042 | 0.892 | 0.375 | 0.625 |
| 0.9 | 0.85 | 0.5 | 0.65 | 0.585308 | 0.651042 | 0.892 | 0.375 | 0.625 |
| 0.9 | 0.85 | 0.65 | 0.65 | 0.585308 | 0.651042 | 0.892 | 0.375 | 0.625 |

## Tuning Plot

![Safe hierarchical policy tuning](figures/safe_hierarchical_policy_tuning_v1.png)

## Research Interpretation

The tuned policy is selected by balancing known-test usefulness and local unknown safety.

A safer policy should increase the local unknown manual-review rate while avoiding an excessive drop in useful known-test decisions.

This tuning stage shows that hierarchical fallback must be controlled carefully. Coarse fallback is useful for known-class uncertainty, but if it is too permissive it can still accept unknown images.
