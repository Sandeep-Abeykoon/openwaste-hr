# Human Labelling Sheet v1 Report

## Purpose

This report documents the human labelling sheet created from the active learning candidates.

## Summary

| metric | value |
| --- | --- |
| total_labelling_rows | 20 |
| manual_review_candidates | 12 |
| coarse_label_candidates | 4 |
| fine_label_candidates | 4 |

## Output Files

| File | Purpose |
|---|---|
| ml/outputs/metrics/human_labelling_sheet_v1.csv | CSV sheet for human annotation |
| ml/outputs/metrics/human_labelling_instructions_v1.md | labelling guide for the reviewer |

## Sheet Preview

| candidate_rank | sample_id | hierarchical_decision_type | hierarchical_final_label | active_learning_score | human_decision | human_fine_label | human_coarse_label | proposed_new_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | local_000002 | manual_review | manual_review | 0.846766 |  |  |  |  |
| 2 | local_000035 | manual_review | manual_review | 0.839029 |  |  |  |  |
| 3 | local_000027 | manual_review | manual_review | 0.834844 |  |  |  |  |
| 4 | local_000020 | manual_review | manual_review | 0.774892 |  |  |  |  |
| 5 | local_000029 | manual_review | manual_review | 0.75257 |  |  |  |  |
| 6 | local_000006 | manual_review | manual_review | 0.747066 |  |  |  |  |
| 7 | local_000033 | manual_review | manual_review | 0.728483 |  |  |  |  |
| 8 | local_000009 | manual_review | manual_review | 0.722906 |  |  |  |  |
| 9 | local_000039 | manual_review | manual_review | 0.719005 |  |  |  |  |
| 10 | local_000031 | manual_review | manual_review | 0.697844 |  |  |  |  |
| 11 | local_000019 | manual_review | manual_review | 0.680643 |  |  |  |  |
| 12 | local_000023 | manual_review | manual_review | 0.677636 |  |  |  |  |
| 13 | local_000037 | coarse_label | recyclable | 0.556322 |  |  |  |  |
| 14 | local_000022 | coarse_label | recyclable | 0.498591 |  |  |  |  |
| 15 | local_000036 | coarse_label | recyclable | 0.492698 |  |  |  |  |
| 16 | local_000015 | coarse_label | recyclable | 0.480001 |  |  |  |  |
| 17 | local_000038 | fine_label | paper_cardboard | 0.337306 |  |  |  |  |
| 18 | local_000025 | fine_label | plastic | 0.323408 |  |  |  |  |
| 19 | local_000018 | fine_label | plastic | 0.311263 |  |  |  |  |
| 20 | local_000001 | fine_label | plastic | 0.306373 |  |  |  |  |

## Research Interpretation

This stage prepares the active learning candidates for human review.

The reviewed annotations can later be used to identify new local labels, confirm mixed or unclear waste cases, and decide which samples should be added to the next dataset version.
