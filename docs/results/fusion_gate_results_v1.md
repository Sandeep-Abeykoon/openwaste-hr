# Calibrated Multi-Score Fusion Gate Results v1

## Purpose

This report evaluates an extension to the final OpenWaste-HR reject-option system.

The previous final decision policy used a single energy-score threshold. The extension replaces the single-score rule with a calibrated multi-score fusion gate trained to predict whether an input is likely to belong to the known taxonomy.

The aim is to reduce false acceptance of unknown images while maintaining known-image coverage and accepted-known accuracy.

## Base Model

| Item | Value |
|---|---|
| Base model | stage_04_add_trashbox_clean_v1 |
| Fusion model | Logistic Regression with StandardScaler |
| Training data for fusion | known_val + unknown_val |
| Final evaluation data | known_test + unknown_test |
| Known classes | cardboard, glass, metal, paper, plastic |
| Unknown classes | biological, textile |

## Fusion Features

The fusion gate used the following score-level features:

| Feature | Meaning |
|---|---|
| confidence | Raw top-1 softmax confidence |
| temperature_scaled_confidence | Calibrated confidence after temperature scaling |
| max_logit | Maximum output logit |
| energy | Energy score from model logits |
| softmax_margin | Difference between top-1 and top-2 softmax probabilities |
| softmax_entropy | Entropy of the softmax probability distribution |

Class probability columns were available, so softmax margin and entropy were computed directly from the class probability distribution.

## Fusion Model Tuning

The Logistic Regression regularisation parameter C was selected using five-fold stratified cross-validation on the calibration data.

| C | CV AUROC mean | CV AUROC std |
|---:|---:|---:|
| 0.1 | 0.8767 | 0.0120 |
| 1.0 | 0.8774 | 0.0116 |
| 3.0 | 0.8775 | 0.0116 |
| 10.0 | 0.8776 | 0.0115 |

The selected value was:

| Item | Value |
|---|---:|
| Best C | 10.0 |

## Threshold Selection

The threshold was selected on the validation set using this objective:

```text
minimise false acceptance of unknown samples
subject to known coverage >= 0.75
```

Selected validation threshold:

| Item | Value |
|---|---:|
| Fusion knownness threshold | 0.5012 |
| Validation known coverage | 0.7507 |
| Validation unknown rejection rate | 0.8681 |
| Validation false acceptance rate | 0.1319 |
| Validation accepted-known accuracy | 0.9750 |
| Validation AUROC | 0.8785 |

## Final Test Results

| Metric | Value |
|---|---:|
| Known test rows | 3,426 |
| Unknown test rows | 1,660 |
| Known coverage | 0.7627 |
| Known rejection rate | 0.2373 |
| Unknown rejection rate | 0.8536 |
| Unknown acceptance rate / FAR | 0.1464 |
| Accepted-known accuracy | 0.9790 |
| Selective risk | 0.0210 |
| Binary balanced accuracy | 0.8082 |
| AUROC known vs unknown | 0.8793 |

## Comparison with Energy-Only Reject Policy

| Method | Test AUROC | Known coverage | Unknown rejection rate | FAR | Accepted-known accuracy |
|---|---:|---:|---:|---:|---:|
| Energy-only reject policy | 0.8789 | 0.7665 | 0.8500 | 0.1500 | 0.9791 |
| Fusion gate v1 | 0.8793 | 0.7627 | 0.8536 | 0.1464 | 0.9790 |

## Interpretation

The calibrated multi-score fusion gate slightly improved the energy-only reject policy.

Compared with the energy-only policy, the fusion gate:

- increased unknown rejection from 0.8500 to 0.8536,
- reduced false acceptance rate from 0.1500 to 0.1464,
- slightly improved AUROC from 0.8789 to 0.8793,
- maintained almost identical accepted-known accuracy.

The improvement is small, but it is still useful because it shows that combining multiple model-derived uncertainty signals can slightly improve the accept/reject decision without retraining the image classifier.

## Research Meaning

This result strengthens the project because the final OpenWaste-HR system is no longer limited to a single-score reject rule. It now includes a calibrated post-hoc fusion model trained on validation-only known and unknown samples.

However, the small improvement also shows that score-level fusion alone is not enough for a large FAR reduction. A stronger future extension should add a feature-space distance score such as Mahalanobis distance or ViM.

## Status

Fusion Gate v1 is complete.

The current final recommendation is:

| System component | Selected method |
|---|---|
| Known-class classifier | Stage 4 MobileNetV3 model |
| Confidence display | Temperature-scaled confidence |
| Reject-option baseline | Energy score |
| Research extension | Calibrated multi-score fusion gate |
| Stronger future extension | Fusion gate with Mahalanobis or ViM feature-distance score |
