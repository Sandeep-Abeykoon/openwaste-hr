# Reject Baseline Comparison v1

## Purpose

This document compares the first OpenWaste-HR reject-option baselines using both known-test selective classification performance and corrected local unknown/manual-review evaluation.

The aim is to decide which reject method should be used as the current safest baseline before moving to hierarchical coarse/fine fallback.

## Baseline A: Closed-Set Classifier

| Metric            |  Value |
| ----------------- | -----: |
| Test accuracy     | 0.6927 |
| Balanced accuracy | 0.6545 |
| Macro-F1          | 0.6456 |
| Weighted-F1       | 0.7009 |

Interpretation:

The closed-set classifier always predicts one known fine label. It is useful as the first baseline, but it cannot reject unknown, ambiguous, or locally shifted images.

## Baseline B: Confidence-Threshold Reject

Known-test selective performance:

| Metric                  |    Value |
| ----------------------- | -------: |
| Selected threshold      |   0.6400 |
| Test coverage           | 0.682292 |
| Test rejection rate     | 0.317708 |
| Test selective accuracy | 0.770992 |
| Test selective macro-F1 | 0.715164 |

local unknown performance:

| Metric                          |    Value |
| ------------------------------- | -------: |
| Unknown samples                 |       40 |
| Rejected unknown count          |       14 |
| Accepted unknown as known count |       26 |
| Unknown rejection rate          | 0.350000 |
| Unknown false acceptance rate   | 0.650000 |

Interpretation:

Confidence-threshold rejection produced the best corrected local unknown rejection result among the current reject baselines. It rejected 14 out of 40 unknown/manual-review images.

## Baseline C1: Maximum Logit Score Reject

Known-test selective performance:

| Metric                  |    Value |
| ----------------------- | -------: |
| Selected threshold      | 2.185518 |
| Test coverage           | 0.716146 |
| Test rejection rate     | 0.283854 |
| Test selective accuracy | 0.760000 |
| Test selective macro-F1 | 0.723257 |

local unknown performance:

| Metric                          |    Value |
| ------------------------------- | -------: |
| Unknown samples                 |       40 |
| Rejected unknown count          |       11 |
| Accepted unknown as known count |       29 |
| Unknown rejection rate          | 0.275000 |
| Unknown false acceptance rate   | 0.725000 |

Interpretation:

Maximum Logit Score gave the best known-test selective macro-F1, but it rejected fewer corrected local unknown images than the confidence-threshold method.

## Baseline C2: Energy Score Reject

Known-test selective performance:

| Metric                  |     Value |
| ----------------------- | --------: |
| Selected threshold      | -2.614076 |
| Test coverage           |  0.744792 |
| Test rejection rate     |  0.255208 |
| Test selective accuracy |  0.741259 |
| Test selective macro-F1 |  0.705525 |

local unknown performance:

| Metric                          |    Value |
| ------------------------------- | -------: |
| Unknown samples                 |       40 |
| Rejected unknown count          |        8 |
| Accepted unknown as known count |       32 |
| Unknown rejection rate          | 0.200000 |
| Unknown false acceptance rate   | 0.800000 |

Interpretation:

Energy Score performed weakest on the corrected local unknown evaluation. It accepted 32 out of 40 local unknown/manual-review images as known labels.

## Decision

For the next stage, OpenWaste-HR will use confidence-threshold rejection as the current safest reject baseline.

Reason:

| Criterion                                            | Best Current Method  |
| ---------------------------------------------------- | -------------------- |
| Known-test selective macro-F1                        | Maximum Logit Score  |
| Known-test selective accuracy                        | Confidence Threshold |
| Corrected local unknown rejection rate               | Confidence Threshold |
| Lowest corrected local unknown false acceptance rate | Confidence Threshold |

Although Maximum Logit Score gives slightly higher known-test selective macro-F1, Confidence Threshold is selected for the next stage because OpenWaste-HR prioritises safe handling of unknown and local manual-review images.

## Key Research Finding

Simple threshold-based rejection is not enough.

Even the best current method, Confidence Threshold, accepted 26 out of 40 corrected local unknown images as known labels.

This supports the need for the proposed hierarchical decision policy:

1. predict a fine label only when confidence is strong
2. fall back to a coarse category when fine-label confidence is weak but coarse evidence is stable
3. send the image to manual review when both fine and coarse decisions are unsafe

## Next Stage

The next implementation stage is the hierarchical coarse/fine decision policy.

Expected output types:

| Condition                                         | Output        |
| ------------------------------------------------- | ------------- |
| confident fine prediction                         | fine label    |
| uncertain fine prediction but stable coarse group | coarse label  |
| unsafe or unknown input                           | manual_review |
