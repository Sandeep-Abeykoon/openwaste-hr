# Human Correction Label Preparation Summary v1

## Purpose

This stage prepares OpenWaste-HR for human correction labels and active learning v2.

## Why This Is Needed

The best current prototype can still assign a known label to an object outside the training taxonomy.

The image:

```text
ml/data/local_unknown/images/local_000001.jpg
```

was visually identified as a rubber slipper / flip-flop.

This object is outside the current known fine-label taxonomy, but the model predicted a known class with high confidence.

## Confirmed Example

| Sample ID    | Human Observation          | Taxonomy Status                | Recommended Action   |
| ------------ | -------------------------- | ------------------------------ | -------------------- |
| local_000001 | rubber slipper / flip-flop | outside_current_known_taxonomy | keep_as_unknown_test |

## Active Learning Link

The human correction workflow will support the next version of the dataset by deciding whether reviewed samples should be:

* added to an existing class
* kept as unknown test samples
* treated as future class candidates
* recollected due to poor quality
* excluded as duplicates

## Current Importance

This stage documents the bridge between model prediction errors and dataset improvement.

It supports the OpenWaste-HR thesis argument that local unknown evaluation and human-in-the-loop active learning are necessary for real-world waste classification.
