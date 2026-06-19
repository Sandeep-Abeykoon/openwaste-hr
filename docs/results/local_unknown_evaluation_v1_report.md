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
| confidence | 40 | 14 | 26 | 0.35 | 0.65 |
| max_logit | 40 | 11 | 29 | 0.275 | 0.725 |
| energy | 40 | 8 | 32 | 0.2 | 0.8 |

## Accepted Unknown Label Distribution

| method_name | pred_label | accepted_count | accepted_percentage_within_method |
| --- | --- | --- | --- |
| confidence | metal | 9 | 34.62 |
| confidence | paper_cardboard | 7 | 26.92 |
| confidence | plastic | 10 | 38.46 |
| energy | glass | 1 | 3.12 |
| energy | metal | 10 | 31.25 |
| energy | paper_cardboard | 8 | 25.0 |
| energy | plastic | 12 | 37.5 |
| energy | residual | 1 | 3.12 |
| max_logit | glass | 1 | 3.45 |
| max_logit | metal | 8 | 27.59 |
| max_logit | paper_cardboard | 8 | 27.59 |
| max_logit | plastic | 12 | 41.38 |

## Rejection Rate Plot

![Local unknown rejection rates](figures/local_unknown_rejection_rates_v1.png)

## Accepted Label Distribution Plot

![Accepted unknown label distribution](figures/local_unknown_accepted_label_distribution_v1.png)

## Research Interpretation

For unknown evaluation, rejection/manual review is the desired behaviour.

Accepted unknown samples are treated as false accepts because the system forced a local unknown item into a known fine label. This result is important because it tests the actual OpenWaste-HR motivation: avoiding unsafe forced predictions on unknown or locally shifted inputs.
