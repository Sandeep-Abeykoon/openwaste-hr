# Active Learning Retraining Impact Plan v1

## Purpose

This section explains how the impact of manual review and active learning should be evaluated in OpenWaste-HR.

## Current Status

The project has already implemented the active learning preparation workflow:

| Component                           | Status   |
| ----------------------------------- | -------- |
| active learning candidate selection | complete |
| human labelling sheet generation    | complete |
| reviewed-label processing script    | complete |
| reviewed local label seed           | complete |
| active learning v2 dataset planning | complete |

However, the full retraining improvement cycle has not yet been completed.

## Why Retraining Has Not Been Performed Yet

Active learning retraining should only use reviewed samples that genuinely belong to the current known taxonomy.

The reviewed seed example, `local_000001`, was identified as a rubber slipper / flip-flop. This item is outside the current known fine-label taxonomy and was therefore kept as an unknown/future-class candidate.

It should not be used as a known training example.

## Correct Active Learning Workflow

The correct workflow is:

| Stage                     | Description                                                               |
| ------------------------- | ------------------------------------------------------------------------- |
| candidate selection       | select uncertain or unknown-like samples                                  |
| human review              | describe and validate the object                                          |
| reviewed label processing | decide whether the sample belongs to a known class                        |
| dataset split decision    | separate known retraining candidates from unknown/future-class candidates |
| retraining                | fine-tune only with valid reviewed known-class samples                    |
| impact comparison         | compare before and after active learning                                  |

## What Counts as an Active Learning Training Sample

A sample can be used for retraining only if it clearly belongs to one of the known fine labels:

* paper_cardboard
* plastic
* glass
* metal
* organic
* residual

A sample should not be used for retraining if it is outside the current taxonomy, unclear, or better treated as a future class.

## Before/After Comparison

If enough reviewed known-class samples are collected, the active learning impact should compare:

| System        | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| Baseline C    | expanded public pretrained model before active learning            |
| Baseline C-AL | expanded public model fine-tuned with reviewed known-class samples |

The comparison should report:

| Metric Area                  | Metrics                                              |
| ---------------------------- | ---------------------------------------------------- |
| closed-set known performance | accuracy, balanced accuracy, macro-F1, weighted-F1   |
| safe hierarchical policy     | known coverage and accepted success rate             |
| unknown handling             | local unknown manual-review or rejection rate        |
| data contribution            | number of reviewed known samples used for retraining |

## Thesis-Safe Interpretation

If enough known-class reviewed samples are available, the project can report a real before/after active learning result.

If not enough known-class reviewed samples are available, the project should report that the active learning mechanism was implemented and validated up to reviewed-label processing and dataset planning, while retraining remains future work.

This is still academically acceptable because it avoids contaminating the known training set with outside-taxonomy samples.
