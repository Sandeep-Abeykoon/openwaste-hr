# Baseline Training v1

## Purpose

This document defines the first closed-set baseline training stage for OpenWaste-HR.

The baseline model is important because the final proposed system must be compared against a normal classifier that always forces an input image into one of the known classes.

## Baseline Role

The baseline is not the main novelty of the project.

The baseline is used to answer:

> How well does a normal closed-set waste classifier perform before adding reject option, coarse fallback, unknown detection, and local active learning?

## Dataset

The first baseline uses the TrashNet manifest files:

```text
ml/data/splits/known_train.csv
ml/data/splits/known_val.csv
ml/data/splits/known_test.csv