# Safe Hierarchical Policy Tuning v1 Report

## Purpose

This report tunes the OpenWaste-HR hierarchical decision policy to improve local unknown safety while preserving useful known-test decisions.

## Selected Thresholds

| threshold | value |
| --- | --- |
| fine_confidence_threshold | 0.995 |
| coarse_confidence_threshold | 0.8 |
| coarse_margin_threshold | 0.15 |
| minimum_confidence_for_coarse | 0.9 |

## Known-Test Metrics

| metric | value |
| --- | --- |
| known_total_samples | 384.0 |
| fine_decision_count | 262.0 |
| coarse_fallback_count | 70.0 |
| manual_review_count | 52.0 |
| known_decision_coverage | 0.864583 |
| known_manual_review_rate | 0.135417 |
| hierarchical_success_rate_over_all | 0.830729 |
| hierarchical_success_rate_over_accepted | 0.960843 |

## Local Unknown Metrics

| metric | value |
| --- | --- |
| unknown_total_samples | 40.0 |
| unknown_manual_review_count | 24.0 |
| unknown_fine_accept_count | 6.0 |
| unknown_coarse_accept_count | 10.0 |
| unknown_accepted_count | 16.0 |
| unknown_manual_review_rate | 0.6 |
| unknown_acceptance_rate | 0.4 |

## Decision Distribution

| dataset | decision_type | count | percentage |
| --- | --- | --- | --- |
| known_test | fine_label | 262 | 68.23 |
| known_test | coarse_label | 70 | 18.23 |
| known_test | manual_review | 52 | 13.54 |
| local_unknown | fine_label | 6 | 15.0 |
| local_unknown | coarse_label | 10 | 25.0 |
| local_unknown | manual_review | 24 | 60.0 |

## Top Candidate Policies

| fine_confidence_threshold | coarse_confidence_threshold | coarse_margin_threshold | minimum_confidence_for_coarse | objective_score | known_decision_coverage | hierarchical_success_rate_over_accepted | unknown_manual_review_rate | unknown_acceptance_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.995 | 0.8 | 0.15 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |
| 0.995 | 0.8 | 0.35 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |
| 0.995 | 0.8 | 0.5 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |
| 0.995 | 0.8 | 0.65 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |
| 0.995 | 0.8 | 0.8 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |
| 0.995 | 0.9 | 0.15 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |
| 0.995 | 0.9 | 0.35 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |
| 0.995 | 0.9 | 0.5 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |
| 0.995 | 0.9 | 0.65 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |
| 0.995 | 0.9 | 0.8 | 0.9 | 0.761169 | 0.864583 | 0.960843 | 0.6 | 0.4 |

## Tuning Plot

![Safe hierarchical policy tuning](figures/pretrained_safe_hierarchical_policy_tuning_v1.png)

## Research Interpretation

The tuned policy is selected by balancing known-test usefulness and local unknown safety.

A safer policy should increase the local unknown manual-review rate while avoiding an excessive drop in useful known-test decisions.

This tuning stage shows that hierarchical fallback must be controlled carefully. Coarse fallback is useful for known-class uncertainty, but if it is too permissive it can still accept unknown images.
