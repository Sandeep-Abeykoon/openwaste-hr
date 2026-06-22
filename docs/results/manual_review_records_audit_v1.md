# Manual Review Records Audit v1

## Purpose

This report audits the current manual review records for OpenWaste-HR active learning.

The goal is to decide whether the existing reviewed records are ready for active learning retraining or whether they should remain as unknown/future-class evidence.

## Summary

| Metric | Value |
|---|---:|
| total reviewed records | 1 |
| known train candidates | 0 |
| known evaluation candidates | 0 |
| unknown or future-class candidates | 1 |
| exclude or review again | 0 |
| review needed | 0 |
| invalid training records | 0 |
| retraining ready | false |

## Decision Counts

| Audit Decision | Count |
| --- | --- |
| known_train_candidate | 0 |
| known_eval_candidate | 0 |
| unknown_or_future_candidate | 1 |
| exclude_or_review_again | 0 |
| review_needed | 0 |

## Known Training Candidates by Class

| Fine Label | Known Training Candidate Count |
| --- | --- |
| paper_cardboard | 0 |
| plastic | 0 |
| glass | 0 |
| metal | 0 |
| organic | 0 |
| residual | 0 |

## Reviewed Record Decisions

| Sample ID | Observation | Decision | Reason |
| --- | --- | --- | --- |
| local_000001 | - | unknown_or_future_candidate | The reviewed sample is outside the current known taxonomy or marked as unknown/future-class. |

## Retraining Decision

Active learning retraining is not ready yet. More reviewed known-class samples are required before fine-tuning the expanded public pretrained model.

## Interpretation

Manual review records should only be used for retraining when they clearly belong to the current known taxonomy.

Outside-taxonomy records should not be forced into known training classes. They should remain useful for local unknown evaluation, future-class analysis, and manual-review evidence.

The next step is to complete more manual review records and then rerun this audit.
