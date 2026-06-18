# Data Inspection v1

## Purpose

This document defines the first data inspection stage for OpenWaste-HR.

Before training any model, the project must verify that the manifest paths, labels, splits, and image files are valid. This prevents training on broken paths, unreadable images, wrong labels, or unbalanced splits without noticing.

## Why This Step Matters

OpenWaste-HR is not only a classification application. It is a research project about safer waste classification under realistic data conditions.

Therefore, data inspection must record:

- number of images per split
- number of images per fine label
- number of images per coarse label
- whether image paths exist
- whether images are readable
- image width and height statistics
- class imbalance observations

## Input

The first inspection uses:

```text
ml/data/splits/trashnet_manifest_v1.csv