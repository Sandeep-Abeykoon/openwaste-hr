# Reviewed Human Label Processing v1 Report

## Purpose

This report processes the human labelling sheet and converts it into review statuses and dataset actions.

## Summary

| metric | value |
| --- | --- |
| total_rows | 20 |
| reviewed_rows | 0 |
| pending_review_rows | 20 |
| invalid_review_rows | 0 |
| ready_for_dataset_rows | 0 |
| add_as_known_sample_rows | 0 |
| add_as_new_unknown_candidate_rows | 0 |
| keep_as_mixed_review_sample_rows | 0 |
| keep_for_review_only_rows | 0 |
| remove_from_future_dataset_rows | 0 |
| fix_annotation_rows | 0 |

## Review Status Counts

| review_status | count |
| --- | --- |
| pending_review | 20 |

## Dataset Action Counts

| dataset_action | count |
| --- | --- |
| pending_review | 20 |

## Processed Sheet Preview

| candidate_rank | sample_id | human_decision | human_fine_label | human_coarse_label | proposed_new_label | review_status | dataset_action | review_validation_message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | local_000002 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 2 | local_000035 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 3 | local_000027 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 4 | local_000020 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 5 | local_000029 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 6 | local_000006 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 7 | local_000033 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 8 | local_000009 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 9 | local_000039 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 10 | local_000031 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 11 | local_000019 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 12 | local_000023 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 13 | local_000037 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 14 | local_000022 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 15 | local_000036 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 16 | local_000015 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 17 | local_000038 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 18 | local_000025 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 19 | local_000018 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |
| 20 | local_000001 | nan | nan | nan | nan | pending_review | pending_review | No human decision entered yet. |

## Ready for Dataset Rows

Ready rows: 0

## Research Interpretation

This stage validates and processes human annotations from the active learning workflow.

Since the human labelling sheet has not yet been filled, all 20 selected active learning candidates remain pending review. This is the expected result at this stage.

After human annotations are added, the same script can be rerun to identify samples that are ready to be added to the next dataset version.

