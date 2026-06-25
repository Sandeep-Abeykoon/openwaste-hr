# Mahalanobis-Enhanced Fusion Gate Results v1

## Purpose

This report evaluates the second fusion-gate extension for OpenWaste-HR.

Fusion Gate v1 combined score-level uncertainty signals such as confidence, temperature-scaled confidence, max-logit, energy, softmax margin, and softmax entropy. Fusion Gate v2 extends this by adding a feature-space distance signal: Mahalanobis knownness.

The aim is to test whether combining logit-level, probability-level, and feature-space evidence improves unknown rejection compared with the previous energy-only policy and score-only fusion gate.

## Base Model

| Item | Value |
|---|---|
| Base model | stage_04_add_trashbox_clean_v1 |
| Classifier backbone | MobileNetV3 Large |
| Fusion model | Logistic Regression with StandardScaler |
| Fusion training data | known_val + unknown_val |
| Final evaluation data | known_test + unknown_test |
| Known classes | cardboard, glass, metal, paper, plastic |
| Unknown classes | biological, textile |

## Feature-Space Method

Mahalanobis statistics were fitted using known_train embeddings only.

| Item | Value |
|---|---:|
| Embedding layer | classifier.3 |
| Embedding dimension | 1280 |
| Mahalanobis fit split | known_train |
| Fit rows | 15,958 |
| Covariance shrinkage | 0.0018 |

The Mahalanobis score measures how close an input embedding is to the known-class feature distribution. Lower distance means the image is more similar to known training examples. The script stores this as `mahalanobis_knownness`, where higher values are more known-like.

## Fusion Gate v2 Features

Fusion Gate v2 used the following features:

| Feature | Type |
|---|---|
| confidence | softmax score |
| temperature_scaled_confidence | calibrated softmax score |
| max_logit | logit score |
| energy | logit-derived open-set score |
| softmax_margin | probability margin |
| softmax_entropy | probability uncertainty |
| mahalanobis_knownness | feature-space distance score |

## Logistic Regression Tuning

The Logistic Regression regularisation parameter C was selected using five-fold stratified cross-validation on the validation calibration data.

| C | CV AUROC mean | CV AUROC std |
|---:|---:|---:|
| 0.1 | 0.9227 | 0.0078 |
| 1.0 | 0.9247 | 0.0072 |
| 3.0 | 0.9249 | 0.0069 |
| 10.0 | 0.9250 | 0.0068 |

The selected value was:

| Item | Value |
|---|---:|
| Best C | 10.0 |

## Threshold Selection

The accept/reject threshold was selected on known_val + unknown_val only using the following objective:

```text
minimise false acceptance of unknown samples
subject to known coverage >= 0.75
```

Selected validation threshold:

| Item | Value |
|---|---:|
| Fusion v2 knownness threshold | 0.6315 |
| Validation known coverage | 0.7512 |
| Validation unknown rejection rate | 0.9380 |
| Validation false acceptance rate | 0.0620 |
| Validation accepted-known accuracy | 0.9708 |
| Validation AUROC | 0.9259 |

## Final Test Results

| Metric | Value |
|---|---:|
| Known test rows | 3,426 |
| Unknown test rows | 1,660 |
| Known coverage | 0.7656 |
| Known rejection rate | 0.2344 |
| Unknown rejection rate | 0.9337 |
| Unknown acceptance rate / FAR | 0.0663 |
| Accepted-known accuracy | 0.9752 |
| Selective risk | 0.0248 |
| Binary balanced accuracy | 0.8497 |
| AUROC known vs unknown | 0.9269 |

## Comparison Against Previous Rejectors

| Method | Test AUROC | Known coverage | Unknown rejection rate | FAR | Accepted-known accuracy |
|---|---:|---:|---:|---:|---:|
| Confidence threshold | 0.8498 | 0.6842 | 0.8831 | 0.1169 | 0.9906 |
| Temperature-scaled confidence | 0.8572 | 0.7341 | 0.8530 | 0.1470 | 0.9877 |
| Max-logit score | 0.8782 | 0.7659 | 0.8506 | 0.1494 | 0.9802 |
| Energy score | 0.8789 | 0.7665 | 0.8500 | 0.1500 | 0.9791 |
| Fusion Gate v1 score-only | 0.8793 | 0.7627 | 0.8536 | 0.1464 | 0.9790 |
| Mahalanobis-only | 0.5636 | 0.7607 | 0.3012 | 0.6988 | 0.9068 |
| Fusion Gate v2 + Mahalanobis | 0.9269 | 0.7656 | 0.9337 | 0.0663 | 0.9752 |

## Interpretation

Fusion Gate v2 produced a major improvement over the energy-only reject policy and Fusion Gate v1.

Compared with energy-only rejection:

- AUROC improved from 0.8789 to 0.9269.
- Unknown rejection improved from 0.8500 to 0.9337.
- False acceptance rate reduced from 0.1500 to 0.0663.
- Known coverage stayed almost unchanged, moving from 0.7665 to 0.7656.
- Accepted-known accuracy remained high, changing from 0.9791 to 0.9752.

This means the enhanced fusion gate rejected many more unknown textile and biological samples while preserving almost the same useful known-item coverage.

## Mahalanobis Ablation Finding

Mahalanobis by itself performed poorly as a standalone reject method. It achieved only 0.5636 AUROC and rejected only 30.12% of unknown test samples.

However, when combined with energy, logits, confidence, margin, and entropy inside the fusion model, it became highly useful. This shows that the feature-space score was not reliable alone, but it provided complementary information when combined with other uncertainty signals.

## Final Research Finding

The Mahalanobis-enhanced fusion gate is the strongest reject-option method tested in OpenWaste-HR.

The result supports the extended research claim that safe open-world waste classification benefits from combining multiple uncertainty signals instead of relying on a single confidence or energy threshold.

## Final Recommended Policy

The final OpenWaste-HR decision policy should now be updated to:

| System component | Selected method |
|---|---|
| Known-class classifier | Stage 4 MobileNetV3 model |
| Confidence display | Temperature-scaled confidence |
| Reject-option method | Fusion Gate v2 with Mahalanobis |
| Final accept threshold | 0.6315 |
| Manual review trigger | fusion knownness score < 0.6315 |

## Status

Fusion Gate v2 with Mahalanobis is complete and should replace the earlier energy-only final decision policy as the strongest final research result.
