# OpenWaste-HR Supervisor Handover Pack v1

## Project Title

OpenWaste-HR: Hierarchical Open-Set Waste Classification with Reject/Manual-Review Option and Local Active Learning

## Current Project Message

OpenWaste-HR improves waste classification by combining pretrained image recognition with hierarchical open-set decision-making, safe reject/manual-review behaviour, local unknown evaluation, and human-in-the-loop active learning.

## Problem Addressed

Most waste classification systems behave as closed-set classifiers. They are trained on a fixed set of known waste categories and are usually forced to predict one of those known labels.

This is risky in real-world settings because local waste images may contain objects outside the training taxonomy. A classifier can still produce high confidence even when the object is not represented in the known classes.

OpenWaste-HR addresses this by adding:

| Component                   | Purpose                                                     |
| --------------------------- | ----------------------------------------------------------- |
| hierarchical taxonomy       | supports fine-label and coarse-label decisions              |
| reject/manual-review option | routes uncertain or unsafe cases to review                  |
| local unknown evaluation    | tests behaviour on unfamiliar local objects                 |
| safe policy tuning          | balances known coverage and unknown safety                  |
| active learning             | records human-reviewed local samples for future improvement |

## Best Current System

The best current system is:

```text id="kbnp69"
Pretrained Safe Hierarchical Policy
```

| Item       | Value                                                                        |
| ---------- | ---------------------------------------------------------------------------- |
| Model      | pretrained transfer-learning model                                           |
| Checkpoint | ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt |
| Policy     | pretrained safe hierarchical policy                                          |

Selected thresholds:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

## Best Evaluation Result

| Metric                            |    Value |
| --------------------------------- | -------: |
| Known-test coverage               | 0.864583 |
| Accepted hierarchical reliability | 0.960843 |
| Local unknown manual-review rate  | 0.600000 |
| Local unknown acceptance rate     | 0.400000 |

## Closed-Set Model Improvement

| Model                    | Accuracy | Macro-F1 |
| ------------------------ | -------: | -------: |
| Scratch-trained baseline | 0.692708 | 0.645600 |
| Pretrained baseline      | 0.888000 | 0.851000 |

The pretrained model improves closed-set known-class classification, but OpenWaste-HR shows that accuracy alone is not enough for real-world unknown handling.

## Policy Comparison

| Policy                       | Known Coverage | Accepted Reliability | Local Unknown Manual Review | Local Unknown Acceptance |
| ---------------------------- | -------------: | -------------------: | --------------------------: | -----------------------: |
| Scratch safe hierarchical    |       0.658854 |             0.889328 |                    0.375000 |                 0.625000 |
| Pretrained hierarchical v1   |       0.976562 |             0.957333 |                    0.075000 |                 0.925000 |
| Pretrained safe hierarchical |       0.864583 |             0.960843 |                    0.600000 |                 0.400000 |

The pretrained safe hierarchical policy is selected because it gives the best safety-coverage balance.

## Live Prototype

The project includes a working prototype with:

| Component               | Status   |
| ----------------------- | -------- |
| single-image inference  | complete |
| batch inference         | complete |
| API wrapper             | complete |
| FastAPI backend         | complete |
| frontend demo page      | complete |
| best policy integration | complete |

The active frontend/backend prototype uses the pretrained safe hierarchical policy.

## Live Demo Example

The demo image used was:

```text id="ng3m56"
ml/data/local_unknown/images/local_000001.jpg
```

Human observation:

```text id="hw3d7n"
rubber slipper / flip-flop
```

This object is outside the current known fine-label taxonomy.

The model result was:

| Field                  | Value           |
| ---------------------- | --------------- |
| Predicted fine label   | paper_cardboard |
| Max softmax confidence | 0.993654        |
| Final decision type    | coarse_label    |
| Final label            | recyclable      |
| Final confidence       | 0.999999        |

This example is useful because it shows why closed-set confidence alone is unsafe for real-world waste classification.

## Active Learning v2 Result

The human-reviewed local seed was assigned the following role:

| Field                   | Value                                   |
| ----------------------- | --------------------------------------- |
| Sample ID               | local_000001                            |
| Human observation       | rubber slipper / flip-flop              |
| Taxonomy status         | outside_current_known_taxonomy          |
| Recommended action      | keep_as_unknown_test                    |
| Active learning v2 role | unknown_test_and_future_class_candidate |

This prevents the unknown object from being wrongly added to known training data.

## Key Thesis Files

| Purpose                       | File                                                            |
| ----------------------------- | --------------------------------------------------------------- |
| methodology chapter           | docs/thesis/methodology_chapter_consolidated_v1.md              |
| implementation chapter        | docs/thesis/implementation_chapter_draft_v1.md                  |
| evaluation chapter draft      | docs/thesis/evaluation_chapter_draft_v1.md                      |
| final evaluation update       | docs/thesis/evaluation_best_policy_active_learning_update_v1.md |
| active learning v2 section    | docs/thesis/active_learning_v2_section_v1.md                    |
| thesis assembly checklist     | docs/thesis/thesis_assembly_checklist_v1.md                     |
| final model/policy comparison | docs/results/final_model_policy_comparison_v1.md                |
| final evaluation summary      | docs/results/final_evaluation_summary_best_policy_v1.md         |

## Completed Pipeline Status

| Stage                                      | Status   |
| ------------------------------------------ | -------- |
| taxonomy and label map                     | complete |
| dataset manifest validation                | complete |
| TrashNet intake                            | complete |
| scratch baseline training                  | complete |
| reject-option baselines                    | complete |
| local unknown evaluation                   | complete |
| hierarchical policy                        | complete |
| safe policy tuning                         | complete |
| pretrained training                        | complete |
| pretrained reject/local unknown evaluation | complete |
| best policy selection                      | complete |
| backend/frontend prototype                 | complete |
| human correction seed                      | complete |
| active learning v2 dataset plan            | complete |
| thesis section updates                     | complete |

## Remaining Work

Recommended next work:

1. review the remaining 19 active-learning candidate images
2. add more local unknown images
3. add more public datasets beyond the current baseline source
4. evaluate more pretrained architectures
5. fine-tune using reviewed active-learning data
6. improve frontend upload support
7. prepare final dissertation formatting and citations

## Supervisor Review Focus

The supervisor can review:

| Area            | Question                                                               |
| --------------- | ---------------------------------------------------------------------- |
| novelty         | Is the hierarchical open-set framing clear enough?                     |
| methodology     | Are the decision-policy thresholds justified clearly?                  |
| evaluation      | Are the policy comparisons explained well?                             |
| active learning | Is the human-in-the-loop workflow acceptable for final-year scope?     |
| prototype       | Is the backend/frontend demo sufficient as proof of implementation?    |
| thesis writing  | Which generated sections should be merged into the final dissertation? |
