# Baseline Evaluation v1

## Purpose

This document defines how the first closed-set TrashNet baseline is evaluated.

The baseline model is a normal forced-choice classifier. It always predicts one of the known labels available in the TrashNet baseline data.

## Why This Baseline Matters

OpenWaste-HR needs a closed-set baseline so that later experiments can compare safer prediction methods against a standard classifier.

The baseline answers:

> What happens when the system is forced to classify every image into a known waste class?

Later stages will compare this against:

1. confidence-threshold rejection
2. max-logit or energy-based rejection
3. hierarchical coarse/fine fallback
4. unknown/manual-review output
5. active-learning adaptation

## Evaluation Data

The first baseline is evaluated using:

```text
ml/data/splits/known_test.csv