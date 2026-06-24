# OpenWaste-HR Final Results Summary v1

## Purpose

This document summarises the final OpenWaste-HR experimental results.

It brings together the main results from:

1. staged dataset expansion,
2. active learning,
3. closed-set waste classification,
4. reject-option open-set evaluation,
5. temperature scaling,
6. calibrated fusion gate,
7. Mahalanobis-enhanced Fusion Gate v2,
8. final single-image inference behaviour.

This report is intended to provide a single thesis-ready summary of the final research outcome.

---

## Final Known Taxonomy

The final classifier was trained on five known recyclable classes:

| Class |
|---|
| cardboard |
| glass |
| metal |
| paper |
| plastic |

These classes are mapped to the coarse category:

```text
recyclable
```

The unknown classes used for open-set validation and testing were:

| Unknown class |
|---|
| biological |
| textile |

The unknown classes were not trained as a sixth class. They were used only for reject-option thresholding and final open-set evaluation.

---

## Final Dataset Summary

| Split | Rows |
|---|---:|
| known_train | 15,958 |
| known_val | 3,417 |
| known_test | 3,426 |
| unknown_val | 1,660 |
| unknown_test | 1,660 |

The final Stage 4 dataset combined the cleaned known-class portions of:

| Dataset |
|---|
| TrashNet |
| RealWaste |
| Garbage Classification V2 |
| TrashBox |

---

## Staged Closed-Set Results

| Stage | Description | Test Accuracy | Test Macro-F1 | Test Balanced Accuracy |
|---|---|---:|---:|---:|
| Stage 1 | TrashNet baseline | 0.9324 | 0.9320 | 0.9357 |
| Stage 2 | TrashNet + RealWaste | 0.9432 | 0.9447 | 0.9445 |
| Stage 3 | Add Garbage V2 | 0.9445 | 0.9445 | 0.9437 |
| Stage 3 + AL | Add active-learning samples | 0.9508 | 0.9509 | 0.9510 |
| Stage 4 | Add cleaned TrashBox | 0.9212 | 0.9213 | 0.9217 |

---

## Final Stage 4 Per-Dataset Results

| Dataset | Rows | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|
| TrashNet | 355 | 0.9634 | 0.9636 | 0.9639 |
| RealWaste | 472 | 0.9089 | 0.9111 | 0.9101 |
| Garbage V2 | 1,082 | 0.9538 | 0.9532 | 0.9551 |
| TrashBox | 1,517 | 0.8919 | 0.8916 | 0.8915 |
| Overall | 3,426 | 0.9212 | 0.9213 | 0.9217 |

---

## Active Learning Findings

Active learning was used between staged dataset expansions to select uncertain or informative external samples for human review.

| Comparison | External Dataset | Baseline Macro-F1 | Active Learning Macro-F1 | Result |
|---|---|---:|---:|---|
| Stage 1 to Stage 2 | RealWaste | 0.5161 | 0.7606 | Improved |
| Stage 2 to Stage 3 | Garbage V2 | 0.7494 | 0.8540 | Improved |
| Stage 3 to Stage 4 | TrashBox | 0.6292 | 0.6901 | Improved |

The results show that active learning improved adaptation to new dataset sources before full staged expansion.

---

## Calibration Result

Temperature scaling was applied to improve confidence calibration.

| Metric | Before Temperature Scaling | After Temperature Scaling |
|---|---:|---:|
| Validation ECE | 0.0397 | 0.0118 |
| Test ECE | 0.0430 | 0.0084 |

The selected temperature was:

```text
1.8482154607772827
```

Temperature-scaled confidence is used for confidence display, while the final accept/reject decision is made by Fusion Gate v2.

---

## Reject-Option and Fusion Comparison

| Method | Test AUROC | Known Coverage | Unknown Rejection | FAR | Accepted-Known Accuracy |
|---|---:|---:|---:|---:|---:|
| Confidence threshold | 0.8498 | 0.6842 | 0.8831 | 0.1169 | 0.9906 |
| Temperature-scaled confidence | 0.8572 | 0.7341 | 0.8530 | 0.1470 | 0.9877 |
| Max-logit score | 0.8782 | 0.7659 | 0.8506 | 0.1494 | 0.9802 |
| Energy score | 0.8789 | 0.7665 | 0.8500 | 0.1500 | 0.9791 |
| Fusion Gate v1 score-only | 0.8793 | 0.7627 | 0.8536 | 0.1464 | 0.9790 |
| Mahalanobis-only | 0.5636 | 0.7607 | 0.3012 | 0.6988 | 0.9068 |
| Fusion Gate v2 + Mahalanobis | 0.9269 | 0.7656 | 0.9337 | 0.0663 | 0.9752 |

---

## Final Selected Reject-Option Policy

The final selected policy is:

```text
Fusion Gate v2 with Mahalanobis feature-distance
```

Final threshold:

```text
0.6314586412215439
```

Decision rule:

```text
if fusion_knownness_score >= 0.6314586412215439:
    accept predicted known class
else:
    send to manual review
```

---

## Improvement Over Energy-Only Baseline

| Metric | Energy-Only | Fusion Gate v2 + Mahalanobis | Change |
|---|---:|---:|---:|
| AUROC | 0.8789 | 0.9269 | +0.0480 |
| Known coverage | 0.7665 | 0.7656 | -0.0009 |
| Unknown rejection | 0.8500 | 0.9337 | +0.0837 |
| FAR | 0.1500 | 0.0663 | -0.0837 |
| Accepted-known accuracy | 0.9791 | 0.9752 | -0.0039 |

The final Fusion Gate v2 policy substantially reduced false acceptance of unknown samples while keeping known coverage almost unchanged.

---

## Statistical Evaluation of Final Fusion Gate v2

To strengthen the final evaluation, the final Fusion Gate v2 policy was also evaluated using bootstrap confidence intervals, calibration metrics, and low-FAR partial AUROC.

### Bootstrap 95% Confidence Intervals

The confidence intervals were computed using 1,000 bootstrap resamples of the final known-test and unknown-test predictions.

| Metric | Point Estimate | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|
| Known coverage | 0.7656 | 0.7522 | 0.7799 |
| Unknown rejection rate | 0.9337 | 0.9217 | 0.9452 |
| False acceptance rate | 0.0663 | 0.0548 | 0.0783 |
| Accepted-known accuracy | 0.9752 | 0.9694 | 0.9809 |
| AUROC known vs unknown | 0.9269 | 0.9197 | 0.9347 |

### Fusion Gate Calibration

Fusion Gate v2 outputs a knownness score between 0 and 1. Its calibration was evaluated using ECE and Brier score.

| Metric | Value |
|---|---:|
| Fusion Gate v2 ECE | 0.0641 |
| Fusion Gate v2 Brier score | 0.1087 |

### Low-FAR Partial AUROC

Because the project prioritises reducing false acceptance of unknown waste, partial AUROC was also reported in low-FAR regions.

| Region | Standardized pAUC |
|---|---:|
| FAR <= 0.05 | 0.8062 |
| FAR <= 0.10 | 0.8421 |

These results show that the final Fusion Gate v2 result is supported by statistical uncertainty estimates, calibration analysis, and safety-focused low-FAR evaluation.

---

## Final Inference Behaviour

### Known Plastic Example

| Field | Value |
|---|---|
| Image | ml/data/raw/trashbox/plastic/plastic 1777.jpg |
| Internal prediction | plastic |
| Fusion knownness score | 0.9969 |
| Threshold | 0.6315 |
| Decision | known_fine_label |
| User-visible label | plastic |
| Coarse label | recyclable |

User-facing message:

```text
This item is likely plastic. It belongs to the recyclable category.
```

### Unknown Textile Example

| Field | Value |
|---|---|
| Image | ml/data/raw/garbage_v2/clothes/clothes_319.jpg |
| Actual unknown type | textile / clothes |
| Internal prediction | paper |
| Fusion knownness score | 0.0713 |
| Threshold | 0.6315 |
| Decision | unknown_manual_review |
| User-visible label | manual_review_required |
| Internal prediction shown to user | false |

User-facing message:

```text
The system is not confident that this item belongs to the supported recyclable classes. Please send it for manual review.
```

---

## Final Research Contribution

The final OpenWaste-HR system demonstrates that safer waste classification should not rely only on closed-set accuracy.

The strongest final system combines:

1. a staged multi-source known-class classifier,
2. active-learning-supported dataset expansion,
3. calibrated confidence display,
4. open-set reject-option evaluation,
5. score-level uncertainty features,
6. feature-space Mahalanobis evidence,
7. a calibrated Fusion Gate v2 accept/reject policy.

The final result shows that combining multiple uncertainty signals can significantly reduce false acceptance of unknown waste items while preserving useful known-class performance.

---

## Final Status

The final selected OpenWaste-HR decision policy is complete.

| Component | Final Selection |
|---|---|
| Classifier | Stage 4 MobileNetV3 |
| Known taxonomy | cardboard, glass, metal, paper, plastic |
| Unknown test classes | biological, textile |
| Confidence display | temperature-scaled confidence |
| Reject-option method | Fusion Gate v2 with Mahalanobis |
| Final threshold | 0.6315 |
| Final unknown action | manual review |
| Final AUROC | 0.9269 |
| Final unknown rejection | 0.9337 |
| Final FAR | 0.0663 |
| FAR 95% CI | 0.0548 to 0.0783 |
| Unknown rejection 95% CI | 0.9217 to 0.9452 |
| AUROC 95% CI | 0.9197 to 0.9347 |
| Fusion Gate v2 ECE | 0.0641 |
| Fusion Gate v2 Brier score | 0.1087 |
| pAUC, FAR <= 0.05 | 0.8062 |
| pAUC, FAR <= 0.10 | 0.8421 |
