# Manual Review Records Completion Plan v1

## Purpose

This document defines how manual review records should be completed and used in the OpenWaste-HR active learning workflow.

The purpose is to prevent reviewed local samples from being incorrectly added into known training classes.

## Why Manual Review Is Needed

OpenWaste-HR is not designed as a normal closed-set classifier. In real-world use, the system may receive objects that are outside the known taxonomy.

Manual review records are needed to decide whether a sample should be:

| Review Outcome             | Meaning                                                                   |
| -------------------------- | ------------------------------------------------------------------------- |
| known training candidate   | the item clearly belongs to an existing known fine label                  |
| known evaluation candidate | the item belongs to a known class but should be used only for evaluation  |
| unknown test candidate     | the item is outside the current taxonomy and should test unknown handling |
| future class candidate     | the item is outside the current taxonomy but may become a new class later |
| exclude or review again    | the item is unclear, duplicated, unreadable, or unsafe to label           |

## Current Known Fine Labels

The current known fine labels are:

| Fine Label      | Coarse Label |
| --------------- | ------------ |
| paper_cardboard | recyclable   |
| plastic         | recyclable   |
| glass           | recyclable   |
| metal           | recyclable   |
| organic         | organic      |
| residual        | residual     |

Only samples that clearly fit one of these fine labels should be used for retraining.

## Manual Review Decision Rules

| Human Observation                                                                                | Recommended Action                                               |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| paper, cardboard, paper packaging                                                                | known_train_candidate or known_eval_candidate as paper_cardboard |
| plastic bottle, plastic wrapper, plastic packaging                                               | known_train_candidate or known_eval_candidate as plastic         |
| glass bottle, glass jar, broken glass item                                                       | known_train_candidate or known_eval_candidate as glass           |
| metal can, metal lid, metal object                                                               | known_train_candidate or known_eval_candidate as metal           |
| food waste, plant matter, compostable organic item                                               | known_train_candidate or known_eval_candidate as organic         |
| mixed dirty trash, non-recyclable residual item                                                  | known_train_candidate or known_eval_candidate as residual        |
| rubber slipper, textile, shoe, cloth, toy, cable, electronic item, battery, unknown local object | unknown_test_candidate or future_class_candidate                 |
| blurry image, duplicate image, unclear object                                                    | exclude_or_review_again                                          |

## Important Data Quality Rule

A reviewed local sample must not be forced into a known class just to increase the training set size.

For example:

| Sample       | Human Observation          | Correct Treatment                      |
| ------------ | -------------------------- | -------------------------------------- |
| local_000001 | rubber slipper / flip-flop | keep as unknown/future-class candidate |

This sample should not be added as paper_cardboard, plastic, residual, or any other existing known class.

## Required Manual Review Fields

Each reviewed sample should include:

| Field                 | Purpose                                                  |
| --------------------- | -------------------------------------------------------- |
| sample_id             | unique sample identifier                                 |
| image_path            | path to the reviewed image                               |
| model_predicted_label | model prediction before review                           |
| model_decision_type   | fine, coarse, or manual review                           |
| human_observation     | short human description of the object                    |
| taxonomy_status       | whether the object belongs to the current known taxonomy |
| reviewed_fine_label   | known fine label if applicable                           |
| reviewed_coarse_label | known coarse label if applicable                         |
| recommended_action    | training, evaluation, unknown, future class, or exclude  |
| reviewer_notes        | short justification                                      |

## Active Learning Use of Reviewed Records

Reviewed records should be separated into two groups:

| Group                                     | Use                                                            |
| ----------------------------------------- | -------------------------------------------------------------- |
| reviewed known-class samples              | can be used for active learning retraining                     |
| reviewed unknown/outside-taxonomy samples | should be used for unknown evaluation or future-class analysis |

## Retraining Eligibility Rule

Active learning retraining should only proceed if there are enough reviewed known-class samples.

Minimum recommended requirement:

| Requirement                                |    Value |
| ------------------------------------------ | -------: |
| minimum reviewed known-class samples       |       10 |
| minimum classes represented                |        2 |
| no outside-taxonomy samples added as known | required |

If these requirements are not met, the project should report active learning as a completed workflow preparation stage rather than claiming retraining improvement.

## Impact Measurement Plan

After active learning retraining, the project should compare:

| Comparison    | Meaning                                                            |
| ------------- | ------------------------------------------------------------------ |
| Baseline C    | expanded public pretrained model before active learning            |
| Baseline C-AL | expanded public pretrained model after active learning fine-tuning |

The impact should be measured using:

| Area                      | Metric                                                                    |
| ------------------------- | ------------------------------------------------------------------------- |
| known classification      | accuracy, balanced accuracy, macro-F1, weighted-F1                        |
| safe policy usefulness    | known coverage                                                            |
| accepted decision quality | hierarchical success rate over accepted decisions                         |
| unknown safety            | local unknown manual review rate or unknown rejection rate                |
| data quality              | number of reviewed samples added to known training versus kept as unknown |

## Research Meaning

Manual review records are not only labels. They are evidence for local adaptation.

The active learning part of OpenWaste-HR should show that uncertain or unknown-like samples can be reviewed by a human, separated safely, and then used either for retraining or for unknown/future-class evaluation.

This protects the model from label contamination and supports the thesis argument that real-world waste classification needs human-in-the-loop dataset improvement.
