# Thesis Assembly Checklist Summary v1

## Purpose

This stage creates a final checklist for assembling the OpenWaste-HR dissertation.

## Current Best Thesis Message

```text id="u9um2k"
OpenWaste-HR improves waste classification by combining pretrained image recognition with hierarchical open-set decision-making, safe reject/manual-review behaviour, local unknown evaluation, and human-in-the-loop active learning.
```

## Best Current System

| Item                             | Value                               |
| -------------------------------- | ----------------------------------- |
| Model                            | pretrained transfer-learning model  |
| Policy                           | pretrained safe hierarchical policy |
| Known-test coverage              | 0.864583                            |
| Accepted reliability             | 0.960843                            |
| Local unknown manual-review rate | 0.600000                            |
| Local unknown acceptance rate    | 0.400000                            |

## Main Thesis Files

| Chapter          | Main Source                                                                                                    |
| ---------------- | -------------------------------------------------------------------------------------------------------------- |
| Methodology      | docs/thesis/methodology_chapter_consolidated_v1.md                                                             |
| Implementation   | docs/thesis/implementation_chapter_draft_v1.md                                                                 |
| Evaluation       | docs/thesis/evaluation_chapter_draft_v1.md and docs/thesis/evaluation_best_policy_active_learning_update_v1.md |
| Active Learning  | docs/thesis/active_learning_v2_section_v1.md                                                                   |
| Final Comparison | docs/results/final_model_policy_comparison_v1.md                                                               |

## Key Example

```text id="etjcxx"
local_000001 = rubber slipper / flip-flop
model prediction = paper_cardboard
human status = outside_current_known_taxonomy
active learning v2 role = unknown_test_and_future_class_candidate
```

## Next Stage

After this checklist, the next stage should create a supervisor handover pack or final dissertation insertion plan.
