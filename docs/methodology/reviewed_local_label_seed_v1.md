# Reviewed Local Label Seed v1

## Purpose

This stage creates the first reviewed local-label seed for OpenWaste-HR active learning v2.

The human-labelling sheet contains 20 active-learning candidate samples. At this stage, only one sample has been confidently reviewed by a human:

```text
local_000001
```

This sample was visually identified as a rubber slipper / flip-flop.

## Why Only One Label Is Seeded

The remaining local unknown samples should not be labelled without visually checking the images.

Therefore, this stage does not invent labels for the full human-labelling sheet. It only records the confirmed human observation for local_000001 and leaves the other rows pending for future review.

## Confirmed Human Observation

| Field                 | Value                                         |
| --------------------- | --------------------------------------------- |
| sample_id             | local_000001                                  |
| image_path            | ml/data/local_unknown/images/local_000001.jpg |
| human_observed_object | rubber slipper / flip-flop                    |
| human_taxonomy_status | outside_current_known_taxonomy                |
| recommended_action    | keep_as_unknown_test                          |
| proposed_new_label    | rubber_slipper_flip_flop                      |

## Model Behaviour on This Image

The best pretrained safe policy predicted a known class/coarse category for this object, even though the object is outside the current known taxonomy.

| Field                  | Value           |
| ---------------------- | --------------- |
| predicted fine label   | paper_cardboard |
| max softmax confidence | 0.993654        |
| final decision type    | coarse_label    |
| final label            | recyclable      |
| final confidence       | 0.999999        |

## Research Meaning

This reviewed seed is important because it captures a real open-world limitation.

A rubber slipper is not represented in the current TrashNet-style known labels, but the model still produced a high-confidence known prediction. This supports the OpenWaste-HR argument that waste classification systems need local unknown evaluation, hierarchical fallback decisions, manual-review routing, and active-learning feedback.

## Outputs

This stage creates:

| Output                                     | Purpose                                           |
| ------------------------------------------ | ------------------------------------------------- |
| human_labelling_sheet_v1_seeded_review.csv | working review sheet with local_000001 filled     |
| reviewed_local_labels_seed_v1.csv          | reviewed local seed record for active learning v2 |
| reviewed_local_labels_seed_summary_v1.json | machine-readable summary                          |
| reviewed_local_labels_seed_v1_report.md    | thesis-friendly report                            |

## Next Step

After this seed is created, the next stage can prepare an active-learning v2 dataset plan. The current seed should be kept as an unknown-test or future-class candidate, not forced into the existing known classes.
