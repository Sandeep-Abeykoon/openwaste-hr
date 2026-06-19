# Local Unknown Evaluation v1

## Purpose

This stage evaluates whether OpenWaste-HR can reject local unknown or difficult images instead of forcing them into known TrashNet-derived fine labels.

The local unknown images are phone-captured images that are not part of the closed-set baseline training data.

## Evaluation Goal

For local unknown images:

| System Behaviour | Interpretation |
|---|---|
| manual_review / rejected | desirable behaviour |
| accepted as known fine label | unsafe forced prediction |

## Evaluated Methods

This stage evaluates three reject methods:

1. Softmax confidence threshold
2. Maximum Logit Score threshold
3. Energy Score threshold

## Threshold Rule

Thresholds are not selected using the local unknown data.

The thresholds are reused from the previous validation-based experiments:

- confidence threshold from validation split
- max-logit threshold from validation split
- energy-score threshold from validation split

This prevents tuning thresholds directly on the unknown test set.

## Metrics

This stage reports:

- total unknown samples
- rejected unknown count
- accepted unknown count
- unknown rejection rate
- unknown false acceptance rate
- accepted known-label distribution

## Research Meaning

This is the first true unknown/manual-review evaluation in OpenWaste-HR.

The result shows whether the current reject-option baselines can route local unknown images to manual review. This is closer to the actual research problem than known-class accuracy alone.