# OpenWaste-HR Thesis Assembly Checklist v1

## Purpose

This checklist shows how the completed OpenWaste-HR materials should be assembled into the final dissertation.

The project has now produced the main methodology, implementation, evaluation, prototype, and active learning materials. This checklist helps decide which files should be inserted into each thesis chapter.

## Current Best Project Message

The strongest current thesis message is:

```text id="lvy8zn"
OpenWaste-HR improves waste classification by combining pretrained image recognition with hierarchical open-set decision-making, safe reject/manual-review behaviour, local unknown evaluation, and human-in-the-loop active learning.
```

## Best Current System

| Item                              | Value                               |
| --------------------------------- | ----------------------------------- |
| Model                             | pretrained transfer-learning model  |
| Decision policy                   | pretrained safe hierarchical policy |
| Known-test coverage               | 0.864583                            |
| Accepted hierarchical reliability | 0.960843                            |
| Local unknown manual-review rate  | 0.600000                            |
| Local unknown acceptance rate     | 0.400000                            |

## Chapter 1: Introduction

Use this chapter to explain the real-world problem.

Include:

| Topic                         | Source Material                             |
| ----------------------------- | ------------------------------------------- |
| waste classification problem  | project proposal and problem background     |
| dataset fragmentation problem | methodology chapter and dataset intake plan |
| closed-set limitation         | local unknown evaluation results            |
| OpenWaste-HR aim              | final evaluation best policy update         |
| contribution summary          | final model and policy comparison           |

Recommended files:

```text id="lvxbq6"
docs/results/final_model_policy_comparison_v1.md
docs/results/final_evaluation_summary_best_policy_v1.md
docs/thesis/evaluation_best_policy_active_learning_update_v1.md
```

## Chapter 2: Literature Review

Use this chapter to discuss existing waste classification datasets, CNN/transfer learning, open-set recognition, uncertainty, and active learning.

Recommended content areas:

| Area                       | What to Discuss                                              |
| -------------------------- | ------------------------------------------------------------ |
| waste image classification | CNNs and pretrained models                                   |
| dataset limitations        | TrashNet-style dataset limitations and missing local objects |
| open-set recognition       | why unknown inputs are difficult                             |
| confidence limitations     | high softmax confidence on unknown objects                   |
| active learning            | human review for uncertain/unknown samples                   |

Recommended project-support files:

```text id="hu24z5"
docs/references/reading_log.md
docs/methodology/dataset_intake_plan_v1.md
docs/methodology/unknown_protocol_v1.md
docs/thesis/active_learning_v2_section_v1.md
```

## Chapter 3: Methodology

Use the consolidated methodology as the main source.

Recommended file:

```text id="ht39fv"
docs/thesis/methodology_chapter_consolidated_v1.md
```

Also include the following method-specific sections where useful:

| Method Area                  | Source File                                            |
| ---------------------------- | ------------------------------------------------------ |
| taxonomy                     | docs/methodology/taxonomy_v1.md                        |
| unknown protocol             | docs/methodology/unknown_protocol_v1.md                |
| dataset intake               | docs/methodology/dataset_intake_plan_v1.md             |
| local unknown dataset        | docs/methodology/local_unknown_dataset_protocol_v1.md  |
| baseline training            | docs/methodology/baseline_training_v1.md               |
| confidence reject            | docs/methodology/confidence_threshold_reject_v1.md     |
| open-set scores              | docs/methodology/open_set_score_baseline_v1.md         |
| hierarchical decision policy | docs/methodology/hierarchical_decision_policy_v1.md    |
| safe policy tuning           | docs/methodology/safe_hierarchical_policy_tuning_v1.md |
| pretrained training          | docs/methodology/pretrained_training_v1.md             |
| active learning v2           | docs/thesis/active_learning_v2_section_v1.md           |

## Chapter 4: Implementation

Use the implementation chapter draft as the main source.

Recommended file:

```text id="332g1j"
docs/thesis/implementation_chapter_draft_v1.md
```

Recommended supporting files:

| Implementation Area        | Source File                                                    |
| -------------------------- | -------------------------------------------------------------- |
| architecture diagram       | docs/architecture/openwaste_hr_architecture_v1.mmd             |
| architecture notes         | docs/architecture/openwaste_hr_architecture_notes_v1.md        |
| inference pipeline         | docs/methodology/inference_pipeline_v1.md                      |
| batch inference            | docs/methodology/batch_inference_pipeline_v1.md                |
| API wrapper                | docs/methodology/prototype_api_wrapper_v1.md                   |
| backend endpoint           | docs/methodology/backend_inference_endpoint_v1.md              |
| frontend demo              | docs/methodology/frontend_demo_v1.md                           |
| best policy integration    | docs/methodology/best_pretrained_safe_policy_integration_v1.md |
| backend/frontend demo test | docs/methodology/backend_frontend_best_policy_demo_test_v1.md  |

## Chapter 5: Evaluation and Results

Use the evaluation chapter draft and the final evaluation update together.

Recommended files:

```text id="9h6edm"
docs/thesis/evaluation_chapter_draft_v1.md
docs/thesis/evaluation_best_policy_active_learning_update_v1.md
docs/results/final_evaluation_summary_best_policy_v1.md
docs/results/final_model_policy_comparison_v1.md
```

Important results to include:

| Result Area                         | Best Source                                                                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| scratch baseline                    | docs/results/baseline_trashnet_v1_report.md                                                                                      |
| confidence reject                   | docs/results/confidence_reject_baseline_v1_report.md                                                                             |
| open-set scores                     | docs/results/open_set_score_baseline_v1_report.md                                                                                |
| local unknown evaluation            | docs/results/local_unknown_evaluation_v1_report.md                                                                               |
| hierarchical policy                 | docs/results/hierarchical_decision_policy_v1_report.md                                                                           |
| safe hierarchical tuning            | docs/results/safe_hierarchical_policy_tuning_v1_report.md                                                                        |
| pretrained baseline                 | docs/results/pretrained_trashnet_v1_report.md                                                                                    |
| pretrained reject options           | docs/results/pretrained_confidence_reject_baseline_v1_report.md and docs/results/pretrained_open_set_score_baseline_v1_report.md |
| pretrained local unknown            | docs/results/pretrained_local_unknown_evaluation_v1_report.md                                                                    |
| pretrained hierarchical policy      | docs/results/pretrained_hierarchical_decision_policy_v1_report.md                                                                |
| pretrained safe hierarchical policy | docs/results/pretrained_safe_hierarchical_policy_tuning_v1_report.md                                                             |
| final comparison                    | docs/results/final_model_policy_comparison_v1.md                                                                                 |
| active learning v2 plan             | docs/results/active_learning_v2_dataset_plan_v1_report.md                                                                        |

## Chapter 6: Discussion

The discussion should focus on what the results mean.

Key points:

| Discussion Point                                  | Evidence                                                    |
| ------------------------------------------------- | ----------------------------------------------------------- |
| pretrained learning improves known classification | accuracy improved from 0.692708 to 0.888000                 |
| high accuracy is not enough                       | unknown objects can still receive high confidence           |
| safe policy improves unknown handling             | local unknown manual-review rate reached 0.600000           |
| hierarchical fallback is useful                   | model can output fine_label, coarse_label, or manual_review |
| active learning is needed                         | local_000001 became unknown-test and future class candidate |

Important example:

```text id="x6n9jp"
local_000001 = rubber slipper / flip-flop
model prediction = paper_cardboard with high confidence
final decision = coarse_label recyclable
human decision = outside_current_known_taxonomy
active learning v2 role = unknown_test_and_future_class_candidate
```

## Chapter 7: Conclusion and Future Work

Use this chapter to summarise the project contribution and next steps.

Conclusion points:

| Point           | Message                                           |
| --------------- | ------------------------------------------------- |
| best model      | pretrained transfer-learning model                |
| best policy     | pretrained safe hierarchical policy               |
| key novelty     | hierarchical open-set decision workflow           |
| prototype       | backend/frontend demo integrated with best policy |
| active learning | human-in-the-loop dataset improvement workflow    |

Future work:

1. review all remaining local unknown candidates
2. expand the local unknown dataset
3. add more public waste datasets
4. evaluate more pretrained architectures
5. fine-tune with reviewed active-learning data
6. add a larger future-class taxonomy
7. improve frontend upload support
8. deploy the backend/frontend prototype

## Figures and Tables Checklist

Recommended figures:

| Figure                             | Source                                             |
| ---------------------------------- | -------------------------------------------------- |
| architecture diagram               | docs/architecture/openwaste_hr_architecture_v1.mmd |
| baseline confusion matrix          | docs/results/figures or ml/outputs/figures         |
| confidence reject curves           | ml/outputs/figures                                 |
| local unknown rejection rates      | ml/outputs/figures                                 |
| hierarchical decision distribution | ml/outputs/figures                                 |
| safe policy tuning plot            | ml/outputs/figures                                 |
| active learning candidate scores   | ml/outputs/figures                                 |
| frontend screenshot                | manual screenshot from live demo                   |

Recommended tables:

| Table                       | Source                                                  |
| --------------------------- | ------------------------------------------------------- |
| taxonomy table              | docs/methodology/taxonomy_v1.md                         |
| dataset manifest summary    | docs/methodology/dataset_intake_plan_v1.md              |
| closed-set comparison       | docs/results/final_model_policy_comparison_v1.md        |
| reject option comparison    | docs/results/final_model_policy_comparison_v1.md        |
| policy comparison           | docs/results/final_model_policy_comparison_v1.md        |
| best policy metrics         | docs/results/final_evaluation_summary_best_policy_v1.md |
| active learning v2 decision | docs/thesis/active_learning_v2_section_v1.md            |

## Final Assembly Order

Recommended order for thesis writing:

1. update Chapter 1 Introduction using the final project message
2. finalise Chapter 2 Literature Review with open-set and active learning sources
3. insert consolidated methodology into Chapter 3
4. insert implementation chapter draft into Chapter 4
5. merge evaluation chapter draft with final evaluation update into Chapter 5
6. write Discussion using the best-policy comparison
7. write Conclusion and Future Work
8. insert figures and tables
9. check citations
10. check consistency of terminology

## Terminology to Keep Consistent

Use these terms consistently:

| Use This Term                       | Avoid                     |
| ----------------------------------- | ------------------------- |
| local unknown dataset               | corrected unknown dataset |
| manual_review                       | reject only               |
| coarse fallback                     | wrong broad prediction    |
| pretrained safe hierarchical policy | final model only          |
| human-in-the-loop active learning   | manual checking only      |
| outside_current_known_taxonomy      | wrong label               |

## Current Status

The project now has a complete v1 research pipeline:

| Stage                                      | Status   |
| ------------------------------------------ | -------- |
| taxonomy                                   | complete |
| dataset manifest                           | complete |
| scratch baseline                           | complete |
| reject-option baselines                    | complete |
| local unknown evaluation                   | complete |
| hierarchical policy                        | complete |
| safe policy tuning                         | complete |
| pretrained baseline                        | complete |
| pretrained reject/local unknown evaluation | complete |
| best policy selection                      | complete |
| backend/frontend prototype                 | complete |
| human correction seed                      | complete |
| active learning v2 plan                    | complete |
| thesis section updates                     | complete |
