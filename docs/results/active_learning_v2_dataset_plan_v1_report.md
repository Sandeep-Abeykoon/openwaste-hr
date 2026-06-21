# Active Learning v2 Dataset Plan v1 Report

## Purpose

This report records the first active learning v2 dataset plan for OpenWaste-HR.

## Summary

| metric | value |
| --- | ---: |
| total_reviewed_seed_rows | 1 |
| known_training_candidates | 0 |
| unknown_test_candidates | 1 |
| future_class_candidates | 1 |
| recollection_candidates | 0 |
| excluded_candidates | 0 |

## Active Learning v2 Role Counts

| role | count |
| --- | ---: |
| unknown_test_and_future_class_candidate | 1 |

## Planned Samples

| sample_id | human_observed_object | human_taxonomy_status | active_learning_v2_role |
| --- | --- | --- | --- |
| local_000001 | rubber slipper / flip-flop | outside_current_known_taxonomy | unknown_test_and_future_class_candidate |

## Research Interpretation

The reviewed local seed is treated as an unknown-test and future-class candidate rather than being forced into an existing known class.

This supports the OpenWaste-HR aim of safer open-set waste classification with human-in-the-loop active learning.
