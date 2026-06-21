# Active Learning v2 Dataset Plan v1

## Purpose

This stage creates the first active learning v2 dataset plan for OpenWaste-HR.

The plan uses the reviewed local-label seed created from the human-reviewed local unknown sample:

```text id="xrydga"
local_000001
```

This sample was visually identified as a rubber slipper / flip-flop and marked as outside the current known taxonomy.

## Input

This stage uses:

```text id="flvlr9"
ml/data/splits/reviewed_local_labels_seed_v1.csv
```

The reviewed seed contains one confirmed local unknown object.

## Confirmed Reviewed Seed

| Field                 | Value                          |
| --------------------- | ------------------------------ |
| sample_id             | local_000001                   |
| human_observed_object | rubber slipper / flip-flop     |
| human_taxonomy_status | outside_current_known_taxonomy |
| recommended_action    | keep_as_unknown_test           |
| proposed_new_label    | rubber_slipper_flip_flop       |

## Why This Plan Is Needed

Active learning should not simply add every reviewed image into the known training set.

If a reviewed image is outside the current known taxonomy, adding it to an existing known class would create label noise.

Therefore, this stage separates possible active learning outcomes into clear dataset roles.

## Dataset Role Rules

| Human Review Result                    | Dataset v2 Role                 |
| -------------------------------------- | ------------------------------- |
| clearly belongs to current known class | add_to_known_training_candidate |
| outside current known taxonomy         | keep_as_unknown_test            |
| possible new class                     | future_class_candidate          |
| unclear image                          | recollection_candidate          |
| duplicate or unusable                  | exclude_from_dataset_v2         |

## Rule Applied to local_000001

The reviewed sample local_000001 is a rubber slipper / flip-flop.

This object is outside the current known taxonomy, so it should not be added to the existing known training classes.

The correct active learning v2 plan is:

| Field                             | Value                                   |
| --------------------------------- | --------------------------------------- |
| include_in_known_training_v2      | false                                   |
| include_in_unknown_test_v2        | true                                    |
| include_as_future_class_candidate | true                                    |
| active_learning_v2_role           | unknown_test_and_future_class_candidate |

## Outputs

This stage creates:

| Output                                          | Purpose                                      |
| ----------------------------------------------- | -------------------------------------------- |
| active_learning_v2_dataset_plan_v1.csv          | dataset role plan for reviewed local samples |
| active_learning_v2_dataset_plan_summary_v1.json | machine-readable summary                     |
| active_learning_v2_dataset_plan_v1_report.md    | thesis-friendly report                       |

## Research Meaning

This stage supports the active-learning part of OpenWaste-HR.

It shows that human feedback is not only used to correct labels. It is also used to decide whether a sample should strengthen the current taxonomy, remain as an unknown-test item, or become a future class candidate.

This is important for open-set waste classification because real local objects may fall outside the original public dataset classes.
