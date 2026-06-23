# Reject-Option and Open-Set Evaluation Results v1

## Purpose

This report evaluates the final Stage 4 model as an uncertainty-aware waste classifier.

The model was trained only on five known classes:

- cardboard
- glass
- metal
- paper
- plastic

The unknown/open-set classes were held out from known-class training:

- biological
- textile

The goal is to test whether the model can reject images that do not belong to the five known classes.

## Evaluation Setup

- Final model: stage_04_add_trashbox_clean_v1
- Known validation rows: 3,417
- Known test rows: 3,426
- Unknown validation rows: 1,660
- Unknown test rows: 1,660
- Unknown labels: biological, textile

Thresholds were selected using known validation and unknown validation data only. Final open-set performance was then reported on known test and unknown test data.

## Calibration Results

Temperature scaling was fitted using known validation labels only.

| Metric | Before temperature scaling | After temperature scaling |
|---|---:|---:|
| Known validation ECE | 0.0397 | 0.0118 |
| Known test ECE | 0.0430 | 0.0084 |

Temperature scaling substantially improved calibration. The known test expected calibration error decreased from 0.0430 to 0.0084.

## Reject-Option Method Comparison

| Method | Test AUROC | Test known coverage | Test unknown rejection rate | Test accepted-known accuracy | Test selective risk |
|---|---:|---:|---:|---:|---:|
| Confidence threshold | 0.8498 | 0.6842 | 0.8831 | 0.9906 | 0.0094 |
| Temperature-scaled confidence | 0.8572 | 0.7341 | 0.8530 | 0.9877 | 0.0123 |
| Max-logit score | 0.8782 | 0.7659 | 0.8506 | 0.9802 | 0.0198 |
| Energy score | 0.8789 | 0.7665 | 0.8500 | 0.9791 | 0.0209 |

## Selected Thresholds

| Method | Direction | Threshold selected from validation |
|---|---|---:|
| Confidence threshold | higher is known | 0.9983 |
| Temperature-scaled confidence | higher is known | 0.9405 |
| Max-logit score | higher is known | 7.1738 |
| Energy score | lower is known | -7.1776 |

## Interpretation

The energy score achieved the best overall known-versus-unknown AUROC on the test set, with an AUROC of 0.8789. Max-logit was very close, with an AUROC of 0.8782.

Confidence thresholding rejected the highest proportion of unknown samples, with an unknown rejection rate of 0.8831. However, it also accepted only 0.6842 of known samples, which means it rejected many valid known-class images.

Temperature-scaled confidence improved calibration strongly and also improved AUROC compared with raw confidence. The known test ECE decreased from 0.0430 to 0.0084 after temperature scaling. However, temperature-scaled confidence still performed below energy and max-logit for known-versus-unknown separation.

Energy score gave the best overall open-set separation and the best known coverage among the tested open-set methods. Therefore, energy score is the strongest final reject-option method for this experiment.

## Practical Decision Policy

For the final system, the recommended safe decision policy is:

1. Predict one of the five known classes.
2. Compute the energy score.
3. Accept the fine label if the energy score is below the selected threshold.
4. If the energy score is above the threshold, reject the image as unknown/manual review.
5. Show the user a safe fallback message instead of forcing an incorrect known-class prediction.

Recommended final threshold:

| Policy item | Value |
|---|---:|
| Reject method | Energy score |
| Threshold | -7.1776 |
| Known test coverage | 0.7665 |
| Unknown test rejection rate | 0.8500 |
| Accepted-known accuracy | 0.9791 |

## Research Finding

The final Stage 4 model achieved strong known-class performance, but the reject-option evaluation shows that open-set handling is still necessary. Without a reject option, unknown textile and biological images would always be forced into one of the known classes.

Using energy scoring, the system rejected 85.00% of unknown test images while retaining 76.65% known test coverage. Among accepted known samples, accuracy remained high at 97.91%.

This supports the OpenWaste-HR research claim that waste classification systems should include uncertainty-aware rejection rather than operating as closed-set classifiers only.

## Status

The reject-option/open-set evaluation is complete for:

- raw confidence thresholding
- temperature-scaled confidence thresholding
- max-logit scoring
- energy scoring

## Next Step

The next step is to create visual evaluation outputs:

- coverage-risk curve
- score distribution plot for known vs unknown
- calibration/ECE plot
- final decision-policy diagram

These figures can then be used in the thesis evaluation chapter.
