# Active Learning v2 Thesis Section Summary v1

## Purpose

This stage creates a thesis-ready section explaining the active learning v2 workflow.

## Main Thesis Point

OpenWaste-HR is not only a normal waste classifier. It includes a human-in-the-loop active learning process for reviewing uncertain or unfamiliar local samples.

## Reviewed Example

| Field              | Value                                   |
| ------------------ | --------------------------------------- |
| sample_id          | local_000001                            |
| human observation  | rubber slipper / flip-flop              |
| taxonomy status    | outside_current_known_taxonomy          |
| recommended action | keep_as_unknown_test                    |
| future role        | unknown_test_and_future_class_candidate |

## Why This Example Is Useful

The model predicted a known label with high confidence, but the object was outside the current known taxonomy.

This supports the thesis argument that closed-set confidence is not enough for real-world waste classification.

## Dataset Decision

The reviewed sample is:

| Role                   | Decision |
| ---------------------- | -------- |
| known training sample  | no       |
| unknown-test sample    | yes      |
| future class candidate | yes      |

## Contribution Link

This section supports the OpenWaste-HR contribution by explaining:

* local unknown evaluation
* hierarchical open-set classification
* manual-review routing
* human-in-the-loop active learning
* future taxonomy expansion

## Next Stage

The next stage should update the thesis evaluation chapter and final prototype summary so that the active learning v2 result is included in the overall project story.
