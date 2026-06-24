# Fusion Gate v2 Statistical Evaluation v1

## Purpose

This report strengthens the final OpenWaste-HR evaluation by adding statistical and calibration analysis for the final Fusion Gate v2 decision policy.

The additional evaluation includes:

1. bootstrap 95% confidence intervals,
2. Fusion Gate v2 calibration using ECE and Brier score,
3. partial AUROC in low false-acceptance-rate regions.

The final Fusion Gate v2 policy uses:

```text
Stage 4 MobileNetV3 classifier
+ temperature-scaled confidence
+ Mahalanobis feature-distance
+ Fusion Gate v2 knownness score
```

## Evaluation Data

| Split | Rows |
|---|---:|
| known_test | 3,426 |
| unknown_test | 1,660 |

The fixed Fusion Gate v2 threshold is:

```text
0.6314586412215439
```

## Point Metrics

| Metric | Value |
|---|---:|
| Known coverage | 0.7656 |
| Known rejection rate | 0.2344 |
| Unknown rejection rate | 0.9337 |
| False acceptance rate | 0.0663 |
| Accepted-known accuracy | 0.9752 |
| Selective risk | 0.0248 |
| Binary balanced accuracy | 0.8497 |
| AUROC known vs unknown | 0.9269 |
| Standardized pAUC, FAR <= 0.05 | 0.8062 |
| Standardized pAUC, FAR <= 0.10 | 0.8421 |
| Brier score | 0.1087 |
| ECE | 0.0641 |

## Bootstrap 95% Confidence Intervals

The confidence intervals were computed using 1,000 bootstrap resamples of the final known-test and unknown-test predictions.

| Metric | Point Estimate | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|
| Known coverage | 0.7656 | 0.7522 | 0.7799 |
| Known rejection rate | 0.2344 | 0.2201 | 0.2478 |
| Unknown rejection rate | 0.9337 | 0.9217 | 0.9452 |
| False acceptance rate | 0.0663 | 0.0548 | 0.0783 |
| Accepted-known accuracy | 0.9752 | 0.9694 | 0.9809 |
| Selective risk | 0.0248 | 0.0191 | 0.0306 |
| Binary balanced accuracy | 0.8497 | 0.8406 | 0.8592 |
| AUROC known vs unknown | 0.9269 | 0.9197 | 0.9347 |
| Standardized pAUC, FAR <= 0.05 | 0.8062 | 0.7877 | 0.8244 |
| Standardized pAUC, FAR <= 0.10 | 0.8421 | 0.8279 | 0.8560 |
| Brier score | 0.1087 | 0.1026 | 0.1139 |
| ECE | 0.0641 | 0.0575 | 0.0709 |

## Calibration Evaluation

Fusion Gate v2 outputs a knownness score between 0 and 1. Since this score is used as the final accept/reject signal, calibration was evaluated using Brier score and expected calibration error.

| Calibration Metric | Value |
|---|---:|
| Brier score | 0.1087 |
| ECE | 0.0641 |
| ECE bins | 10 |

The Brier score measures the quality of the probabilistic knownness estimates. The ECE value measures the gap between predicted knownness and the empirical known rate across confidence bins.

The final ECE of 0.0641 indicates moderate calibration of the Fusion Gate v2 knownness score. This is acceptable for the current prototype because the system uses a fixed validation-selected threshold rather than relying only on the raw probability as a user-facing confidence value.

## Partial AUROC at Low FAR

Since the project focuses on safer unknown rejection, partial AUROC was computed in low false-acceptance-rate regions.

| Region | Standardized pAUC |
|---|---:|
| FAR <= 0.05 | 0.8062 |
| FAR <= 0.10 | 0.8421 |

These results show that Fusion Gate v2 maintains useful separation between known and unknown samples even in safety-critical low-FAR regions.

## Interpretation

The statistical evaluation supports the robustness of the final Fusion Gate v2 result.

The key final metric, false acceptance rate, was:

```text
0.0663
```

with a 95% bootstrap confidence interval of:

```text
[0.0548, 0.0783]
```

The unknown rejection rate was:

```text
0.9337
```

with a 95% bootstrap confidence interval of:

```text
[0.9217, 0.9452]
```

This confirms that the final system consistently rejects most unknown textile and biological samples while maintaining useful known-class coverage.

## Final Research Meaning

This additional evaluation strengthens the final research claim.

The project can now report not only that Fusion Gate v2 improved over the energy-only baseline, but also that the final result is supported by bootstrap confidence intervals, calibration analysis, and low-FAR partial AUROC.

This makes the final evaluation more defensible for a thesis because it shows:

1. the main open-set metrics are stable,
2. the final knownness score has measured calibration quality,
3. the model performs well in low false-acceptance-rate regions, which matter for safer deployment.

## Output Files

| Purpose | File |
|---|---|
| Statistical summary JSON | ml/outputs/fusion_gate/stage_04_fusion_gate_v2_statistical_eval/fusion_gate_v2_statistical_summary_v1.json |
| Statistical metrics CSV | ml/outputs/fusion_gate/stage_04_fusion_gate_v2_statistical_eval/fusion_gate_v2_statistical_metrics_v1.csv |
| ECE bin CSV | ml/outputs/fusion_gate/stage_04_fusion_gate_v2_statistical_eval/fusion_gate_v2_ece_bins_v1.csv |
| Low-FAR ROC curve CSV | ml/outputs/fusion_gate/stage_04_fusion_gate_v2_statistical_eval/fusion_gate_v2_low_far_roc_curve_v1.csv |
| Calibration plot | ml/outputs/fusion_gate/stage_04_fusion_gate_v2_statistical_eval/fusion_gate_v2_calibration_plot_v1.png |
| Low-FAR ROC plot | ml/outputs/fusion_gate/stage_04_fusion_gate_v2_statistical_eval/fusion_gate_v2_low_far_roc_plot_v1.png |

## Status

Fusion Gate v2 statistical evaluation is complete.
