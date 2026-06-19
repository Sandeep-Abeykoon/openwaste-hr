# Reviewed Human Label Processing v1

## Purpose

This stage processes the human labelling sheet created from active learning candidates.

The goal is to convert reviewed human annotations into a clean status file and a dataset-ready file.

## Input

The input is:

* ml/outputs/metrics/human_labelling_sheet_v1.csv

This file contains active learning candidates and empty human annotation fields.

## Output

The processing stage creates:

* reviewed_label_decisions_v1.csv
* reviewed_label_ready_for_dataset_v1.csv
* reviewed_label_status_summary_v1.json
* reviewed_label_processing_v1_report.md

## Review Status

Each row is assigned a review status.

| Status         | Meaning                                                                |
| -------------- | ---------------------------------------------------------------------- |
| pending_review | the human annotation fields are still empty                            |
| reviewed       | a human decision has been entered                                      |
| invalid_review | a human decision exists but required information is missing or invalid |

## Human Decisions

| Human Decision    | Meaning                                          |
| ----------------- | ------------------------------------------------ |
| known_label       | the sample fits an existing fine/coarse label    |
| new_unknown_class | the sample represents a possible new local class |
| mixed_waste       | the image contains mixed objects/materials       |
| unclear_image     | the image is too unclear for reliable annotation |
| remove_sample     | the image should not be used in future data      |

## Dataset Actions

| Dataset Action               | Meaning                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| pending_review               | wait for human annotation                                     |
| add_as_known_sample          | add to future local known dataset                             |
| add_as_new_unknown_candidate | keep as candidate for future new class or open-set evaluation |
| keep_as_mixed_review_sample  | keep for manual-review/mixed-case analysis                    |
| keep_for_review_only         | do not add to training yet                                    |
| remove_from_future_dataset   | exclude from future dataset versions                          |

## Research Meaning

This stage closes the active learning loop at the data-management level.

The system selects useful local samples, prepares them for human review, and then processes the reviewed labels into structured dataset decisions.
