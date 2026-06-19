# Local Unknown Evaluation v1

## Purpose

This report evaluates whether the current OpenWaste-HR reject baselines can route local phone-captured unknown images to manual review.

## Known Labels Used by the Model

paper_cardboard, plastic, glass, metal, residual

## Thresholds Used

Thresholds were selected from validation data in earlier experiments. The local unknown set was not used for threshold selection.

| method_name | score_column | threshold | accept_direction |
| --- | --- | --- | --- |
| confidence | max_softmax_confidence | 0.64 | greater_equal |
| max_logit | max_logit_score | 2.185518 | greater_equal |
| energy | energy_score | -2.614076 | less_equal |

## Unknown Rejection Metrics

| method_name | total_unknown_samples | rejected_unknown_count | accepted_unknown_as_known_count | unknown_rejection_rate | unknown_false_acceptance_rate |
| --- | --- | --- | --- | --- | --- |
| confidence | 42 | 16 | 26 | 0.380952 | 0.619048 |
| max_logit | 42 | 11 | 31 | 0.261905 | 0.738095 |
| energy | 42 | 6 | 36 | 0.142857 | 0.857143 |

## Accepted Unknown Label Distribution

| method_name | pred_label | accepted_count | accepted_percentage_within_method |
| --- | --- | --- | --- |
| confidence | glass | 4 | 15.38 |
| confidence | metal | 17 | 65.38 |
| confidence | paper_cardboard | 1 | 3.85 |
| confidence | plastic | 4 | 15.38 |
| energy | glass | 4 | 11.11 |
| energy | metal | 24 | 66.67 |
| energy | plastic | 7 | 19.44 |
| energy | residual | 1 | 2.78 |
| max_logit | glass | 4 | 12.9 |
| max_logit | metal | 21 | 67.74 |
| max_logit | plastic | 6 | 19.35 |

## Rejection Rate Plot

![Local unknown rejection rates](figures/local_unknown_rejection_rates_v1.png)

## Accepted Label Distribution Plot

![Accepted unknown label distribution](figures/local_unknown_accepted_label_distribution_v1.png)

## Research Interpretation

For unknown evaluation, rejection/manual review is the desired behaviour.

Accepted unknown samples are treated as false accepts because the system forced a local unknown item into a known fine label. This result is important because it tests the actual OpenWaste-HR motivation: avoiding unsafe forced predictions on unknown or locally shifted inputs.
