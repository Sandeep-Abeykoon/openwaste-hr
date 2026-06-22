# Manual Review Working Sheet v1

## Purpose

This report prepares the manual review worksheet for OpenWaste-HR active learning v2.

The working sheet collects candidate images that need human review before active learning retraining can be considered.

## Summary

| Metric | Value |
|---|---:|
| total working sheet rows | 20 |
| pending review rows | 20 |
| already reviewed rows | 0 |

## Output CSV

The working sheet was written to:

ml/outputs/active_learning/manual_review_working_sheet_v1.csv

## Manual Review Fields to Complete

For each pending row, complete these fields:

| Field | Allowed / Expected Value |
|---|---|
| human_observation | short description of the object |
| taxonomy_status | current_known_taxonomy, outside_current_known_taxonomy, or unclear |
| reviewed_fine_label | paper_cardboard, plastic, glass, metal, organic, residual, or blank if unknown |
| reviewed_coarse_label | recyclable, organic, residual, or blank if unknown |
| recommended_action | known_train_candidate, known_eval_candidate, unknown_test_candidate, future_class_candidate, or exclude_or_review_again |
| active_learning_v2_role | known_retraining_candidate, known_evaluation_candidate, unknown_test_and_future_class_candidate, or excluded |
| reviewer_notes | short justification |

## Important Rule

Do not force outside-taxonomy samples into known classes.

Only samples that clearly belong to paper_cardboard, plastic, glass, metal, organic, or residual should be used for known-class retraining.

## Next Step

After the worksheet is completed, rerun the manual review audit.

If enough valid known-class samples exist, active learning retraining can proceed.

If not, the reviewed records should be reported as workflow evidence and unknown/future-class analysis.
