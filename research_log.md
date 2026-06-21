# OpenWaste-HR Research Log

## Project Direction

Selected direction: OpenWaste-HR, a hierarchical open-set waste classification system with reject option, coarse-to-fine fallback, and local active learning.

## Why This Direction Was Selected

The project moves beyond standard closed-set waste classification. Instead of forcing every input into a known class, the system will classify known items, fall back to coarse categories when uncertain, and reject unknown or ambiguous waste items for manual review.

## Initial Research Gap

Most existing waste classification systems are evaluated as closed-set classifiers, while real waste streams are open-world and may contain unknown, mixed, damaged, contaminated, or locally unusual items.

## Initial Implementation Plan

1. Freeze project taxonomy.
2. Prepare dataset mapping files.
3. Build closed-set baseline model.
4. Add confidence/reject-option baseline.
5. Build hierarchical coarse/fine model.
6. Evaluate unknown detection.
7. Add local correction UI.
8. Run active-learning experiment.
9. Export model for deployment.
10. Build final demo and thesis-ready results.

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-06-18 | Selected OpenWaste-HR direction | Stronger novelty than standard classifier + XAI + advice system |
| 2026-06-18 | Repository structure created | Keeps ML, backend, frontend, data, documentation, and experiments organized |
| 2026-06-18 | Frozen Taxonomy v1 | Created 7 fine known classes, 4 coarse classes, and reserved unknown/manual-review labels |
| 2026-06-18 | Added dataset intake plan | Created dataset source plan, manifest template, label mapping template, and validation tests |
| 2026-06-18 | Added TrashNet manifest builder | Created first dataset intake script for folder-based closed-set baseline data |
| 2026-06-18 | Added data inspection pipeline | Created manifest-based image loader, image validation, label distribution summaries, and inspection figures |
| 2026-06-18 | Added baseline training pipeline | Created PyTorch dataset, label encoder, MobileNetV3 baseline model, and closed-set training script |
| 2026-06-19 | Added confidence-threshold reject baseline | Added first reject-option baseline using validation-selected softmax confidence threshold |
| 2026-06-19 | Added max-logit and energy-score reject baselines | Added logit-based reject-option baselines to compare against softmax confidence thresholding |
| 2026-06-19 | Added local unknown dataset protocol | Created local unknown collection protocol, metadata template, and manifest builder for future unknown/manual-review evaluation |
| 2026-06-19 | Ran local unknown evaluation | Evaluated softmax confidence, maximum logit, and energy-score rejection on 40 local phone-captured unknown images |
| 2026-06-19 | Compared corrected reject baselines | Selected confidence-threshold rejection as the safest current baseline before hierarchical fallback |
| 2026-06-19 | Added hierarchical decision policy | Implemented first fine-label, coarse-fallback, and manual-review decision policy |

## Taxonomy v1 Summary

Known fine labels:

1. paper_cardboard
2. plastic
3. glass
4. metal
5. organic
6. e_waste_battery
7. residual

Known coarse labels:

1. recyclable
2. organic
3. hazardous
4. residual

Reserved labels:

- unknown
- manual_review

The unknown label is not used as a normal training class. It is used for open-set evaluation, rejection, manual review, and active learning.

## Dataset Intake v1 Summary

The project will not mix image datasets directly.

Each image must be tracked using a manifest row containing:

- source dataset
- original label
- OpenWaste-HR fine label
- OpenWaste-HR coarse label
- known or unknown status
- experiment usage role
- image path
- license or citation note

Dataset intake starts with a simple baseline source, then moves to harder open-world and local unknown evaluation.

Unknown samples are not used as normal known-class training data in the first baseline.

## TrashNet Intake v1 Summary

TrashNet is used as the first simple closed-set baseline dataset.

TrashNet original labels are mapped into OpenWaste-HR labels as follows:

| TrashNet Label | Fine Label | Coarse Label |
|---|---|---|
| cardboard | paper_cardboard | recyclable |
| paper | paper_cardboard | recyclable |
| plastic | plastic | recyclable |
| glass | glass | recyclable |
| metal | metal | recyclable |
| trash | residual | residual |

The generated manifest files are:

- ml/data/splits/trashnet_manifest_v1.csv
- ml/data/splits/known_train.csv
- ml/data/splits/known_val.csv
- ml/data/splits/known_test.csv

These files prepare the first baseline training stage.

## TrashNet Data Inspection v1 Summary

The first real TrashNet manifest contains:

| Usage | Count |
|---|---:|
| known_train | 1766 |
| known_val | 377 |
| known_test | 384 |

Fine-label distribution:

| Fine Label | Count |
|---|---:|
| paper_cardboard | 997 |
| glass | 501 |
| plastic | 482 |
| metal | 410 |
| residual | 137 |

Research observation:

TrashNet is suitable for the first closed-set baseline because it provides simple folder-based waste images. However, it has important limitations for OpenWaste-HR:

1. It does not contain organic waste as a dedicated class.
2. It does not contain e-waste or battery waste as a dedicated class.
3. The residual/trash class is much smaller than the other main material classes.
4. It does not directly test unknown rejection or local adaptation.

Therefore, TrashNet will be used only for the first baseline pipeline. Later stages must add additional dataset sources and local unknown images.

## Baseline Training v1 Summary

The first training pipeline prepares a closed-set baseline classifier using TrashNet.

This baseline predicts only the fine labels currently available from TrashNet:

1. paper_cardboard
2. plastic
3. glass
4. metal
5. residual

The baseline does not yet include:

1. organic
2. e_waste_battery
3. unknown rejection
4. coarse fallback
5. active learning

This is intentional. The closed-set baseline is needed so that the later OpenWaste-HR method can be compared against a normal forced-classification model.

### Actual Baseline v1 Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.6927 |
| Balanced accuracy | 0.6545 |
| Macro-F1 | 0.6456 |
| Weighted-F1 | 0.7009 |

Training note:

The first official closed-set baseline used MobileNetV3 trained from scratch with class weights and early stopping based on validation macro-F1. The best checkpoint was selected at epoch 34. This baseline is retained as Baseline A. A stronger pretrained baseline may be added later, but the next research step is to evaluate reject-option behaviour.

## Confidence-Threshold Reject Baseline v1 Summary

This stage adds the first reject-option behaviour to OpenWaste-HR.

The closed-set classifier no longer has to force every prediction. Instead:

| Condition | Output |
|---|---|
| confidence >= selected threshold | accept fine-label prediction |
| confidence < selected threshold | manual_review |

The threshold is selected using the validation split only. The test split is used only for final evaluation.

Generated files:

- docs/results/confidence_reject_baseline_v1_report.md
- docs/results/figures/confidence_reject_coverage_risk_v1.png
- docs/results/figures/confidence_reject_confidence_histogram_v1.png
- ml/outputs/metrics/confidence_reject_val_predictions_v1.csv
- ml/outputs/metrics/confidence_reject_test_predictions_v1.csv
- ml/outputs/metrics/confidence_reject_threshold_sweep_v1.csv
- ml/outputs/metrics/confidence_reject_selected_threshold_v1.json
- ml/outputs/metrics/confidence_reject_test_metrics_v1.json

Research note:

This is the first move from closed-set classification toward safer open-world behaviour. It does not yet detect true unknown classes, but it tests whether uncertain known-class predictions can be rejected for manual review.

### Actual Confidence-Reject v1 Metrics

| Metric | Value |
|---|---:|
| Selected threshold | 0.6400 |
| Test coverage | 0.682292 |
| Test rejection rate | 0.317708 |
| Test forced accuracy | 0.692708 |
| Test selective accuracy | 0.770992 |
| Test selective macro-F1 | 0.715164 |

## Open-Set Score Baseline v1 Summary

This stage adds two logit-based reject baselines:

| Method | Score | Accept Rule |
|---|---|---|
| Maximum Logit Score | max(logits) | accept if score >= selected threshold |
| Energy Score | -T * logsumexp(logits / T) | accept if score <= selected threshold |

The threshold for each score is selected using the validation split only. The known test split is used only for final evaluation.

Generated files:

- docs/results/open_set_score_baseline_v1_report.md
- docs/results/figures/open_set_score_coverage_risk_v1.png
- docs/results/figures/open_set_score_histogram_v1.png
- ml/outputs/metrics/open_set_score_val_predictions_v1.csv
- ml/outputs/metrics/open_set_score_test_predictions_v1.csv
- ml/outputs/metrics/open_set_score_threshold_sweep_v1.csv
- ml/outputs/metrics/open_set_score_selected_thresholds_v1.json
- ml/outputs/metrics/open_set_score_test_metrics_v1.json

Research note:

This is a stronger reject-option baseline than softmax confidence because it uses raw classifier logits. At this stage, the evaluation still uses known validation and test images. The next unknown-data step is required before claiming true unknown detection performance.


### Actual Open-Set Score v1 Metrics

| Method              | Selected Threshold | Test Coverage | Test Rejection Rate | Test Selective Accuracy | Test Selective Macro-F1 |
| ------------------- | -----------------: | ------------: | ------------------: | ----------------------: | ----------------------: |
| Maximum Logit Score |           2.185518 |      0.716146 |            0.283854 |                0.760000 |                0.723257 |
| Energy Score        |          -2.614076 |      0.744792 |            0.255208 |                0.741259 |                0.705525 |


## Local Unknown Dataset Protocol v1 Summary

This stage prepares the first local unknown dataset pipeline.

The local unknown dataset will contain safe phone-camera images of difficult, ambiguous, mixed-material, damaged, contaminated, reflective, or locally specific waste items.

These images are not used as normal known-class training data.

They are assigned:

| Field | Value |
|---|---|
| source_dataset | local_phone_images |
| original_label | unknown |
| fine_label | unknown |
| coarse_label | unknown |
| is_known | false |

Allowed usage values:

| Usage | Purpose |
|---|---|
| unknown_test | unknown/manual-review evaluation |
| local_unknown | local difficult sample storage |
| active_learning_candidate | later human-correction/adaptation experiment |

Research note:

This prepares the project for true unknown-item evaluation. The next stage will use the current softmax confidence, maximum logit, and energy-score baselines to test whether local unknown images are correctly routed to manual review.

## Local Unknown Dataset 

The first local image collection included some items that were too close to known TrashNet-trained categories. Therefore, those local unknown evaluation results are not treated as final unknown-detection evidence.

The local unknown dataset was recreated using a corrected 40-image phone-captured dataset. The corrected dataset contains unknown, ambiguous, mixed-material, damaged, contaminated, e-waste-like, textile, rubber, foam, wood, ceramic-like, and manual-review-style local items.

The corrected collection sheet was also checked so that each row matches the same image number, for example local_000001 describes local_000001.jpg.

The corrected dataset is reserved only for unknown/manual-review evaluation and is not used for closed-set training.

# Local Unknown Dataset v1 Summary

| Item | Value |
|---|---:|
| Local unknown images | 40 |
| Usage | unknown_test |
| Fine label | unknown |
| Coarse label | unknown |
| Source dataset | local_phone_images |

Generated files:

- ml/data/local_unknown/images/local_000001.jpg to local_000040.jpg
- ml/data/splits/local_unknown_collection_sheet_v1.csv
- ml/data/splits/local_unknown_manifest_v1.csv
- ml/data/splits/local_unknown_image_renaming_map_v1.csv

Research note:

This corrected local unknown dataset is used to test whether OpenWaste-HR can route unknown or ambiguous local phone-captured images to manual review instead of forcing them into known TrashNet-derived fine labels.
Research note:

This is the first true unknown/manual-review evaluation in the project. It tests the main OpenWaste-HR motivation more directly than closed-set accuracy because the system is evaluated on local unknown images not used in training.

### Local Unknown Evaluation v1 Metrics

| Method              | Unknown Samples | Rejected Unknown Count | Accepted Unknown as Known Count | Unknown Rejection Rate | Unknown False Acceptance Rate |
| ------------------- | --------------: | ---------------------: | ------------------------------: | ---------------------: | ----------------------------: |
| Confidence          |              40 |                     14 |                              26 |               0.350000 |                      0.650000 |
| Maximum Logit Score |              40 |                     11 |                              29 |               0.275000 |                      0.725000 |
| Energy Score        |              40 |                      8 |                              32 |               0.200000 |                      0.800000 |

Interpretation:

This corrected evaluation uses the recreated 40-image local unknown dataset. The previous 42-image local unknown result is not treated as final evidence because the earlier dataset contained some images that were too close to known TrashNet-trained classes.

Among the three reject baselines, the confidence-threshold method produced the strongest unknown rejection result. It rejected 14 out of 40 unknown/manual-review images, giving an unknown rejection rate of 0.350000. Maximum Logit Score rejected 11 out of 40 images, while Energy Score rejected 8 out of 40 images.

However, even the best method still accepted 26 out of 40 unknown images as known labels. This shows that simple threshold-based rejection is not sufficient for reliable open-world deployment. Therefore, the next stage should introduce hierarchical coarse/fine fallback with manual-review logic.

## Corrected Reject Baseline Comparison v1 Summary

The reject baselines were compared using known-test selective performance and the corrected 40-image local unknown/manual-review dataset.

Decision:

Confidence-threshold rejection is selected as the current safest reject baseline before hierarchical fallback.

Reason:

* Maximum Logit Score produced the best known-test selective macro-F1.
* Confidence Threshold produced the best corrected local unknown rejection rate.
* Energy Score performed weakest on the corrected local unknown set.
* The project prioritises safe handling of unknown/local difficult inputs, not only known-test selective accuracy.

Corrected local unknown comparison:

| Method              | Unknown Samples | Rejected Unknown Count | Accepted Unknown as Known Count | Unknown Rejection Rate | Unknown False Acceptance Rate |
| ------------------- | --------------: | ---------------------: | ------------------------------: | ---------------------: | ----------------------------: |
| Confidence          |              40 |                     14 |                              26 |               0.350000 |                      0.650000 |
| Maximum Logit Score |              40 |                     11 |                              29 |               0.275000 |                      0.725000 |
| Energy Score        |              40 |                      8 |                              32 |               0.200000 |                      0.800000 |

Important finding:

Even the best current method still falsely accepted 26 out of 40 corrected local unknown images as known labels. This supports the need for the next stage: hierarchical coarse/fine fallback with manual-review logic.

## Hierarchical Decision Policy v1 Summary

This stage adds the first OpenWaste-HR hierarchical decision policy.

The policy can output:

| Decision Type | Meaning |
|---|---|
| fine_label | confident fine-grained prediction |
| coarse_label | broader fallback category when fine prediction is uncertain |
| manual_review | unsafe, ambiguous, or low-confidence input |

Policy v1 thresholds:

| Threshold | Value |
|---|---:|
| fine_confidence_threshold | 0.64 |
| coarse_confidence_threshold | 0.80 |
| coarse_margin_threshold | 0.15 |
| minimum_confidence_for_coarse | 0.35 |

Generated files:

- docs/results/hierarchical_decision_policy_v1_report.md
- docs/results/figures/hierarchical_decision_distribution_v1.png
- ml/outputs/metrics/hierarchical_known_test_decisions_v1.csv
- ml/outputs/metrics/hierarchical_local_unknown_decisions_v1.csv
- ml/outputs/metrics/hierarchical_decision_policy_metrics_v1.json
- ml/outputs/metrics/hierarchical_decision_distribution_v1.csv

Research note:

This is the first implementation of the main OpenWaste-HR decision logic. The system no longer has only accept/reject behaviour. It can now return a fine label, fall back to a coarse category, or send the sample to manual review.

### Actual Hierarchical Decision Policy v1 Metrics

Known-test performance:

| Metric                                  |    Value |
| --------------------------------------- | -------: |
| Known total samples                     |      384 |
| Fine decision count                     |      262 |
| Coarse fallback count                   |       96 |
| Manual review count                     |       26 |
| Known decision coverage                 | 0.932292 |
| Known manual review rate                | 0.067708 |
| Fine correct count                      |      202 |
| Coarse correct count                    |       93 |
| Hierarchical success count              |      295 |
| Fine accuracy on fine decisions         | 0.770992 |
| Coarse accuracy on coarse decisions     | 0.968750 |
| Hierarchical success rate over all      | 0.768229 |
| Hierarchical success rate over accepted | 0.824022 |

Local unknown performance:

| Metric                      |    Value |
| --------------------------- | -------: |
| Unknown total samples       |       40 |
| Unknown manual review count |        3 |
| Unknown fine accept count   |       26 |
| Unknown coarse accept count |       11 |
| Unknown accepted count      |       37 |
| Unknown manual review rate  | 0.075000 |
| Unknown acceptance rate     | 0.925000 |

Decision distribution:

| Dataset       | Fine Label | Coarse Label | Manual Review |
| ------------- | ---------: | -----------: | ------------: |
| Known test    |        262 |           96 |            26 |
| Local unknown |         26 |           11 |             3 |

Interpretation:

Hierarchical Decision Policy v1 improved known-test usefulness because it accepted 358 out of 384 known-test images through either fine-label or coarse-label output. This gave a known decision coverage of 0.932292 and a hierarchical success rate of 0.824022 over accepted known-test decisions.

The coarse fallback was highly reliable on known-test samples, with 93 correct coarse decisions out of 96 coarse fallback decisions. This produced a coarse accuracy of 0.968750 on coarse decisions.

However, the local unknown result shows that Policy v1 is still too permissive for open-world use. Only 3 out of 40 local unknown images were routed to manual review, while 37 were still accepted as either fine or coarse outputs. This means the unknown manual review rate was only 0.075000 and the unknown acceptance rate was 0.925000.

Research conclusion:

Policy v1 successfully proves that the system can produce fine-label, coarse-label, and manual-review outputs. However, it is not safe enough for unknown handling because the coarse fallback accepts too many unknown images. The next stage should tune the hierarchical policy so that coarse fallback is allowed for known-class uncertainty but restricted for local unknown or unsafe inputs.

## Safe Hierarchical Policy Tuning v1 Summary

This stage tunes the hierarchical decision policy to improve local unknown safety while preserving useful known-test decisions.

The tuning process searched **480 threshold candidates** using known-test predictions and the local unknown dataset.

### Actual Safe Hierarchical Policy Tuning v1 Metrics

Selected thresholds:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.900000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.650000 |
| minimum_confidence_for_coarse | 0.650000 |

Known-test performance:

| Metric                                  |    Value |
| --------------------------------------- | -------: |
| Known total samples                     |      384 |
| Fine decision count                     |      145 |
| Coarse fallback count                   |      108 |
| Manual review count                     |      131 |
| Known decision coverage                 | 0.658854 |
| Known manual review rate                | 0.341146 |
| Fine correct count                      |      122 |
| Coarse correct count                    |      103 |
| Hierarchical success count              |      225 |
| Fine accuracy on fine decisions         | 0.841379 |
| Hierarchical success rate over all      | 0.585938 |
| Hierarchical success rate over accepted | 0.889328 |

Local unknown performance:

| Metric                      |    Value |
| --------------------------- | -------: |
| Unknown total samples       |       40 |
| Unknown manual review count |       15 |
| Unknown fine accept count   |       16 |
| Unknown coarse accept count |        9 |
| Unknown accepted count      |       25 |
| Unknown manual review rate  | 0.375000 |
| Unknown acceptance rate     | 0.625000 |

Decision distribution:

| Dataset       | Fine Label | Coarse Label | Manual Review |
| ------------- | ---------: | -----------: | ------------: |
| Known test    |        145 |          108 |           131 |
| Local unknown |         16 |            9 |            15 |

Interpretation:

The tuned policy selected stricter thresholds than the first hierarchical policy. The fine confidence threshold increased to 0.900000 and the coarse margin threshold increased to 0.650000.

Compared with the first hierarchical policy, the local unknown manual-review count improved from 3 out of 40 to 15 out of 40. This means the local unknown manual-review rate increased from 0.075000 to 0.375000.

The trade-off is that known-test decision coverage decreased to 0.658854. However, the accepted known-test decisions became more reliable, with a hierarchical success rate over accepted decisions of 0.889328.

Research conclusion:

Safe hierarchical tuning improves local unknown safety but introduces a clear coverage-safety trade-off. The tuned policy is more conservative: it sends more uncertain images to manual review while keeping higher reliability among accepted known-test decisions.

This supports the OpenWaste-HR argument that open-world waste classification should not only optimise accuracy. It must also manage uncertainty, reject unsafe inputs, and use manual review for ambiguous local cases.


## Final Policy Comparison v1 Summary

This stage compares the main OpenWaste-HR decision policies.

Compared methods:

| Method                             | Output Types                               |
| ---------------------------------- | ------------------------------------------ |
| Closed-set baseline                | fine label only                            |
| Confidence-threshold reject        | fine label or manual review                |
| Hierarchical decision policy v1    | fine label, coarse label, or manual review |
| Safe hierarchical policy tuning v1 | fine label, coarse label, or manual review |

Key comparison:

| Method                             | Known Coverage | Local Unknown Manual Review Rate | Main Interpretation                                                        |
| ---------------------------------- | -------------: | -------------------------------: | -------------------------------------------------------------------------- |
| Closed-set baseline                |       1.000000 |                         0.000000 | always predicts a known label                                              |
| Confidence-threshold reject        |       0.682292 |                         0.350000 | best simple reject baseline                                                |
| Hierarchical decision policy v1    |       0.932292 |                         0.075000 | useful for known-test fallback but too permissive for local unknown images |
| Safe hierarchical policy tuning v1 |       0.658854 |                         0.375000 | safest current hierarchical policy                                         |

Final current decision:

Safe hierarchical policy tuning v1 is selected as the current OpenWaste-HR decision policy.

Reason:

* it supports fine-label, coarse-label, and manual-review outputs
* it improves manual-review routing for local unknown images
* it gives high accepted-decision reliability on known-test samples
* it demonstrates the project’s key coverage-safety trade-off

Research conclusion:

OpenWaste-HR should not be evaluated only as a normal waste classifier. The important contribution is uncertainty-aware decision-making: the system can choose between detailed prediction, coarse fallback, and manual review.

## Active Learning Candidate Selection v1 Summary

This stage selects local unknown dataset images for human labelling.

The candidate selector uses the safe hierarchical policy decisions and ranks samples using:

| Factor                    | Purpose                                                   |
| ------------------------- | --------------------------------------------------------- |
| decision priority         | prioritises manual-review and suspicious accepted samples |
| prediction entropy        | prioritises uncertain probability distributions           |
| confidence uncertainty    | prioritises lower-confidence predictions                  |
| coarse margin uncertainty | prioritises unstable coarse decisions                     |

Selection configuration:

| Setting                     | Value |
| --------------------------- | ----: |
| Total local unknown samples |    40 |
| Selected candidates         |    20 |
| Manual-review quota         |    12 |
| Coarse-label quota          |     4 |
| Fine-label quota            |     4 |

Actual candidate summary:

| Metric                     |    Value |
| -------------------------- | -------: |
| Selected candidate count   |       20 |
| Manual-review candidates   |       12 |
| Coarse-label candidates    |        4 |
| Fine-label candidates      |        4 |
| Mean active learning score | 0.616382 |
| Max active learning score  | 0.846766 |
| Min active learning score  | 0.306373 |

Generated files:

* docs/results/active_learning_candidate_selection_v1_report.md
* docs/results/figures/active_learning_candidate_scores_v1.png
* ml/outputs/metrics/active_learning_candidates_v1.csv
* ml/outputs/metrics/active_learning_candidate_summary_v1.json
* ml/outputs/metrics/active_learning_candidate_distribution_v1.csv

Research note:

This stage creates the first local active learning loop. The system identifies which local unknown images should be labelled by a human next, instead of treating all unlabelled images equally.

The selected batch contains 20 candidates: 12 manual-review cases, 4 coarse-label cases, and 4 fine-label cases. This gives a balanced human-labelling batch that includes uncertain samples, broad fallback cases, and suspicious accepted predictions.


## Human Labelling Sheet v1 Summary

This stage creates a human labelling sheet for the selected active learning candidates.

Input:

* ml/outputs/metrics/active_learning_candidates_v1.csv

Output:

* ml/outputs/metrics/human_labelling_sheet_v1.csv
* ml/outputs/metrics/human_labelling_instructions_v1.md
* docs/results/human_labelling_sheet_v1_report.md

The sheet contains 20 active learning candidates and includes empty human annotation columns:

| Human Annotation Column | Purpose                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| human_decision          | reviewer decision such as known label, new class, mixed waste, unclear image, or remove sample |
| human_fine_label        | existing fine label if applicable                                                              |
| human_coarse_label      | broader category if applicable                                                                 |
| proposed_new_label      | new local label suggestion                                                                     |
| human_confidence        | reviewer confidence                                                                            |
| human_notes             | reviewer explanation                                                                           |
| reviewed_by             | reviewer identifier                                                                            |
| review_date             | annotation date                                                                                |

Research note:

This stage prepares the selected active learning candidates for human review. It creates a practical annotation workflow for converting model-identified uncertain and risky local samples into labelled feedback.

## Human Labelling Sheet v1 Summary

This stage creates a human labelling sheet for the selected active learning candidates.

Input:

* ml/outputs/metrics/active_learning_candidates_v1.csv

Outputs:

* ml/outputs/metrics/human_labelling_sheet_v1.csv
* ml/outputs/metrics/human_labelling_instructions_v1.md
* docs/results/human_labelling_sheet_v1_report.md

Actual summary:

| Metric                   | Value |
| ------------------------ | ----: |
| Total labelling rows     |    20 |
| Manual-review candidates |    12 |
| Coarse-label candidates  |     4 |
| Fine-label candidates    |     4 |
| Tests passed             |    73 |

Human annotation columns added:

| Human Annotation Column | Purpose                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| human_decision          | reviewer decision such as known label, new class, mixed waste, unclear image, or remove sample |
| human_fine_label        | existing fine label if applicable                                                              |
| human_coarse_label      | broader category if applicable                                                                 |
| proposed_new_label      | new local label suggestion                                                                     |
| human_confidence        | reviewer confidence                                                                            |
| human_notes             | reviewer explanation                                                                           |
| reviewed_by             | reviewer identifier                                                                            |
| review_date             | annotation date                                                                                |

Research note:

This stage prepares the selected active learning candidates for human review. It creates a practical annotation workflow for converting model-identified uncertain and risky local samples into labelled feedback.

## Reviewed Human Label Processing v1 Summary

This stage processes the human labelling sheet and converts each row into a review status and dataset action.

Input:

* ml/outputs/metrics/human_labelling_sheet_v1.csv

Outputs:

* ml/outputs/metrics/reviewed_label_decisions_v1.csv
* ml/outputs/metrics/reviewed_label_ready_for_dataset_v1.csv
* ml/outputs/metrics/reviewed_label_status_summary_v1.json
* docs/results/reviewed_label_processing_v1_report.md

Expected current status:

| Metric                 | Value |
| ---------------------- | ----: |
| Total rows             |    20 |
| Reviewed rows          |     0 |
| Pending review rows    |    20 |
| Invalid review rows    |     0 |
| Ready for dataset rows |     0 |

Research note:

At this stage the human labelling sheet has been created but not filled. Therefore, all 20 rows are expected to remain pending review. After human annotation is added, the same processing script can be rerun to identify rows that are ready for the next dataset version.

## Reviewed Human Label Processing v1 Summary

This stage processes the human labelling sheet and converts each row into a review status and dataset action.

Input:

* ml/outputs/metrics/human_labelling_sheet_v1.csv

Outputs:

* ml/outputs/metrics/reviewed_label_decisions_v1.csv
* ml/outputs/metrics/reviewed_label_ready_for_dataset_v1.csv
* ml/outputs/metrics/reviewed_label_status_summary_v1.json
* docs/results/reviewed_label_processing_v1_report.md

Actual status:

| Metric                 | Value |
| ---------------------- | ----: |
| Total rows             |    20 |
| Reviewed rows          |     0 |
| Pending review rows    |    20 |
| Invalid review rows    |     0 |
| Ready for dataset rows |     0 |
| Tests passed           |    80 |

Research note:

At this stage the human labelling sheet has been created but not filled. Therefore, all 20 rows are expected to remain pending review. After human annotation is added, the same processing script can be rerun to identify rows that are ready for the next dataset version.

## Inference Pipeline v1 Summary

This stage creates the first single-image inference pipeline for OpenWaste-HR.

The pipeline accepts one image path and returns:

| Output                       | Meaning                                        |
| ---------------------------- | ---------------------------------------------- |
| pred_label                   | model’s top fine-label prediction              |
| max_softmax_confidence       | confidence of the top fine prediction          |
| top_coarse_label             | aggregated coarse-label prediction             |
| top_coarse_confidence        | confidence of the top coarse group             |
| coarse_margin                | separation between top and second coarse group |
| hierarchical_decision_type   | fine_label, coarse_label, or manual_review     |
| hierarchical_final_label     | final OpenWaste-HR output                      |
| hierarchical_decision_reason | explanation of the final decision              |

The inference pipeline uses the selected safe hierarchical policy thresholds:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.900000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.650000 |
| minimum_confidence_for_coarse | 0.650000 |

Actual single-image inference result:

| Field                         | Value                                         |
| ----------------------------- | --------------------------------------------- |
| sample_id                     | local_000001                                  |
| image_path                    | ml/data/local_unknown/images/local_000001.jpg |
| device                        | cuda                                          |
| pred_label                    | plastic                                       |
| max_softmax_confidence        | 0.962933                                      |
| top_coarse_label              | recyclable                                    |
| top_coarse_confidence         | 0.999999                                      |
| coarse_margin                 | 0.999999                                      |
| hierarchical_decision_type    | fine_label                                    |
| hierarchical_final_label      | plastic                                       |
| hierarchical_final_confidence | 0.962933                                      |
| hierarchical_decision_reason  | fine_confidence_high                          |

Fine-label probability output:

| Fine Label      | Probability |
| --------------- | ----------: |
| paper_cardboard |    0.002948 |
| plastic         |    0.962933 |
| glass           |    0.026874 |
| metal           |    0.007244 |
| residual        |    0.000000 |

Generated files:

* ml/src/openwaste_hr/inference/single_image_inference.py
* ml/configs/inference_pipeline.yaml
* docs/methodology/inference_pipeline_v1.md
* ml/outputs/metrics/single_image_inference_result_v1.json
* ml/outputs/metrics/single_image_inference_result_v1.md

Research note:

This stage turns the experimental OpenWaste-HR pipeline into a usable single-image inference component. It demonstrates how a new image can be processed into a final fine-label, coarse-label, or manual-review decision.

The first inference run predicted `plastic` for `local_000001` with high confidence, so the safe hierarchical policy returned a fine-label decision.

## Batch Inference Pipeline v1 Summary

This stage creates the first batch inference pipeline for OpenWaste-HR.

The pipeline processes a folder of images and returns one row per image with:

| Output                       | Meaning                                        |
| ---------------------------- | ---------------------------------------------- |
| pred_label                   | model’s top fine-label prediction              |
| max_softmax_confidence       | confidence of the top fine prediction          |
| top_coarse_label             | aggregated coarse-label prediction             |
| top_coarse_confidence        | confidence of the top coarse group             |
| coarse_margin                | separation between top and second coarse group |
| hierarchical_decision_type   | fine_label, coarse_label, or manual_review     |
| hierarchical_final_label     | final OpenWaste-HR output                      |
| hierarchical_decision_reason | reason for the final decision                  |

Input folder:

* ml/data/local_unknown/images

Generated files:

* ml/src/openwaste_hr/inference/batch_inference.py
* ml/configs/batch_inference_pipeline.yaml
* docs/methodology/batch_inference_pipeline_v1.md
* ml/outputs/metrics/batch_inference_results_v1.csv
* ml/outputs/metrics/batch_inference_summary_v1.json
* ml/outputs/metrics/batch_inference_report_v1.md

Research note:

This stage demonstrates that OpenWaste-HR can process a folder of local images and export structured fine-label, coarse-label, or manual-review decisions. This is useful for later backend integration, manual review, and active learning workflows.


### Actual Batch Inference Pipeline v1 Result

Batch inference was run on the local unknown image folder.

| Metric                  |    Value |
| ----------------------- | -------: |
| Tests passed            |       94 |
| Images processed        |       42 |
| Fine-label decisions    |       16 |
| Coarse-label decisions  |       10 |
| Manual-review decisions |       16 |
| Accepted decisions      |       26 |
| Fine-label rate         | 0.380952 |
| Coarse-label rate       | 0.238095 |
| Manual-review rate      | 0.380952 |
| Accepted rate           | 0.619048 |

Decision distribution:

| Decision Type | Count | Percentage |
| ------------- | ----: | ---------: |
| fine_label    |    16 |      38.10 |
| coarse_label  |    10 |      23.81 |
| manual_review |    16 |      38.10 |

Final label distribution:

| Final Label     | Count | Percentage |
| --------------- | ----: | ---------: |
| manual_review   |    16 |      38.10 |
| recyclable      |    10 |      23.81 |
| plastic         |     6 |      14.29 |
| metal           |     6 |      14.29 |
| paper_cardboard |     4 |       9.52 |

Generated files:

* ml/src/openwaste_hr/inference/batch_inference.py
* ml/configs/batch_inference_pipeline.yaml
* docs/methodology/batch_inference_pipeline_v1.md
* ml/outputs/metrics/batch_inference_results_v1.csv
* ml/outputs/metrics/batch_inference_summary_v1.json
* ml/outputs/metrics/batch_inference_report_v1.md

Research note:

This stage demonstrates that OpenWaste-HR can process a folder of images and export structured fine-label, coarse-label, or manual-review decisions. The batch inference run is a prototype inference workflow and should be reported separately from the earlier 40-sample local unknown evaluation.

## Prototype API Wrapper v1 Summary

This stage creates a lightweight API-style wrapper around the OpenWaste-HR single-image inference pipeline.

The wrapper accepts a request with:

| Field      | Required | Meaning                            |
| ---------- | -------- | ---------------------------------- |
| image_path | yes      | project-relative path to the image |
| sample_id  | no       | optional image/sample identifier   |

The wrapper returns a structured response containing:

| Section             | Meaning                                |
| ------------------- | -------------------------------------- |
| status              | success or error                       |
| request             | sample ID and image path               |
| prediction          | model prediction and confidence values |
| decision            | final OpenWaste-HR decision            |
| class_probabilities | known fine-label probabilities         |
| policy              | safe hierarchical thresholds           |
| metadata            | device and pipeline version            |

Actual prototype API wrapper result:

| Field                  | Value                                         |
| ---------------------- | --------------------------------------------- |
| Tests passed           | 101                                           |
| status                 | success                                       |
| request_id             | demo_request_001                              |
| sample_id              | local_000001                                  |
| image_path             | ml/data/local_unknown/images/local_000001.jpg |
| pred_label             | plastic                                       |
| max_softmax_confidence | 0.962933                                      |
| top_coarse_label       | recyclable                                    |
| top_coarse_confidence  | 0.999999                                      |
| coarse_margin          | 0.999999                                      |
| decision_type          | fine_label                                    |
| final_label            | plastic                                       |
| final_confidence       | 0.962933                                      |
| reason                 | fine_confidence_high                          |
| device                 | cuda                                          |
| pipeline_version       | prototype_api_wrapper_v1                      |

Class probability output:

| Fine Label      | Probability |
| --------------- | ----------: |
| paper_cardboard |    0.002948 |
| plastic         |    0.962933 |
| glass           |    0.026874 |
| metal           |    0.007244 |
| residual        |    0.000000 |

Generated files:

* docs/methodology/prototype_api_wrapper_v1.md
* ml/configs/prototype_api_wrapper.yaml
* ml/src/openwaste_hr/inference/api_wrapper.py
* tests/test_api_wrapper.py
* ml/outputs/metrics/prototype_api_response_v1.json
* ml/outputs/metrics/prototype_api_response_v1.md

Research note:

This stage prepares OpenWaste-HR for backend integration by wrapping the model output in a stable request/response structure. The backend can later call this wrapper instead of directly handling model internals.

## Backend Inference Endpoint Skeleton v1 Summary

This stage creates a lightweight FastAPI backend endpoint for OpenWaste-HR inference.

Created endpoints:

| Method | Endpoint               | Purpose                                 |
| ------ | ---------------------- | --------------------------------------- |
| GET    | /health                | backend health check                    |
| POST   | /api/inference/predict | run OpenWaste-HR inference on one image |

The prediction endpoint accepts:

| Field      | Required | Meaning                     |
| ---------- | -------- | --------------------------- |
| image_path | yes      | project-relative image path |
| sample_id  | no       | optional sample identifier  |
| request_id | no       | optional request identifier |

The endpoint returns the same structured OpenWaste-HR response used by the prototype API wrapper:

| Section             | Meaning                                                   |
| ------------------- | --------------------------------------------------------- |
| status              | success or error                                          |
| request             | sample ID and image path                                  |
| prediction          | model prediction values                                   |
| decision            | final fine-label, coarse-label, or manual-review decision |
| class_probabilities | known fine-label probabilities                            |
| policy              | safe hierarchical thresholds                              |
| metadata            | device and pipeline version                               |

Generated files:

* docs/methodology/backend_inference_endpoint_v1.md
* backend/app/main.py
* backend/app/api/inference_routes.py
* backend/app/schemas/inference_schema.py
* backend/app/services/inference_service.py
* tests/test_backend_inference_endpoint.py

Research note:

This stage moves OpenWaste-HR closer to a usable prototype by exposing the inference workflow through a backend-style endpoint. It prepares the system for later frontend integration and demonstration.

### Actual Backend Inference Endpoint v1 Result

The backend endpoint was tested successfully using FastAPI and Uvicorn.

Test result:

| Metric                     |   Value |
| -------------------------- | ------: |
| Tests passed               |     105 |
| Warnings                   |       1 |
| Health endpoint status     | success |
| Prediction endpoint status | success |

Health endpoint result:

| Field   | Value                         |
| ------- | ----------------------------- |
| status  | ok                            |
| service | openwaste-hr-backend          |
| version | backend_inference_endpoint_v1 |

Prediction endpoint input:

| Field      | Value                                         |
| ---------- | --------------------------------------------- |
| image_path | ml/data/local_unknown/images/local_000001.jpg |
| sample_id  | local_000001                                  |
| request_id | backend_demo_001                              |

Prediction endpoint output:

| Field                  | Value                    |
| ---------------------- | ------------------------ |
| status                 | success                  |
| request_id             | backend_demo_001         |
| pred_label             | plastic                  |
| max_softmax_confidence | 0.962933                 |
| top_coarse_label       | recyclable               |
| top_coarse_confidence  | 0.999999                 |
| coarse_margin          | 0.999999                 |
| decision_type          | fine_label               |
| final_label            | plastic                  |
| final_confidence       | 0.962933                 |
| reason                 | fine_confidence_high     |
| device                 | cuda                     |
| pipeline_version       | prototype_api_wrapper_v1 |

Class probability output:

| Fine Label      | Probability |
| --------------- | ----------: |
| paper_cardboard |    0.002948 |
| plastic         |    0.962933 |
| glass           |    0.026874 |
| metal           |    0.007244 |
| residual        |    0.000000 |

Research note:

This stage successfully exposes the OpenWaste-HR inference workflow through a backend endpoint. The backend can receive an image path, call the inference wrapper, and return a structured prediction and final hierarchical decision. This prepares the project for frontend integration and prototype demonstration.

## Simple Frontend Demo Page v1 Summary

This stage creates a lightweight frontend demo page for OpenWaste-HR.

Created files:

* docs/methodology/frontend_demo_v1.md
* frontend/index.html
* frontend/styles.css
* frontend/app.js
* tests/test_frontend_demo_files.py

Backend update:

* backend/app/main.py was updated with CORS middleware so the browser demo can call the FastAPI endpoint.

The frontend allows a user to enter:

| Field       | Meaning                     |
| ----------- | --------------------------- |
| image_path  | project-relative image path |
| sample_id   | optional sample identifier  |
| backend URL | FastAPI backend URL         |

The frontend displays:

| Output              | Meaning                                    |
| ------------------- | ------------------------------------------ |
| final label         | final OpenWaste-HR label                   |
| decision type       | fine_label, coarse_label, or manual_review |
| decision reason     | explanation for the final decision         |
| predicted label     | model top fine-label prediction            |
| confidence          | model confidence value                     |
| coarse label        | top coarse-level prediction                |
| class probabilities | probability for each known fine label      |
| raw response        | full backend JSON response                 |

Research note:

This stage creates the first visible OpenWaste-HR prototype demo. The user can run the backend, open the frontend page, send an image path, and view the final hierarchical decision in the browser.

### Actual Simple Frontend Demo Page v1 Result

The frontend demo page was tested successfully with the FastAPI backend.

Test result:

| Metric                  |                  Value |
| ----------------------- | ---------------------: |
| Tests passed            |                    110 |
| Warnings                |                      1 |
| Frontend request status |                success |
| Backend endpoint called | /api/inference/predict |

Frontend input:

| Field       | Value                                         |
| ----------- | --------------------------------------------- |
| image_path  | ml/data/local_unknown/images/local_000001.jpg |
| sample_id   | local_000001                                  |
| backend URL | http://127.0.0.1:8000                         |

Frontend output:

| Field                  | Value                       |
| ---------------------- | --------------------------- |
| status                 | success                     |
| request_id             | frontend_demo_1782022315233 |
| pred_label             | plastic                     |
| max_softmax_confidence | 0.962933                    |
| top_coarse_label       | recyclable                  |
| top_coarse_confidence  | 0.999999                    |
| coarse_margin          | 0.999999                    |
| decision_type          | fine_label                  |
| final_label            | plastic                     |
| final_confidence       | 0.962933                    |
| reason                 | fine_confidence_high        |
| device                 | cuda                        |
| pipeline_version       | prototype_api_wrapper_v1    |

Class probability output:

| Fine Label      | Probability |
| --------------- | ----------: |
| paper_cardboard |    0.002948 |
| plastic         |    0.962933 |
| glass           |    0.026874 |
| metal           |    0.007244 |
| residual        |    0.000000 |

Research note:

This stage successfully creates a visible OpenWaste-HR prototype demo. The frontend sends an image path to the backend, the backend calls the inference pipeline, and the browser displays the final hierarchical decision, model confidence, class probabilities, and raw JSON response.

## Prototype Run Guide and Demo Checklist v1 Summary

This stage creates the project run guide and supervisor demo checklist.

Created files:

* docs/methodology/prototype_run_guide_v1.md
* docs/supervisor_updates/prototype_demo_checklist_v1.md
* tests/test_prototype_run_guide_docs.py

The prototype run guide explains how to:

| Task                       | Purpose                                       |
| -------------------------- | --------------------------------------------- |
| activate the environment   | prepare the project runtime                   |
| run the test suite         | confirm the prototype is stable               |
| check the checkpoint       | confirm trained model weights are available   |
| run single-image inference | test direct inference                         |
| run batch inference        | process a folder of images                    |
| run the API wrapper        | test the backend-friendly response format     |
| start the FastAPI backend  | expose the inference endpoint                 |
| test the backend endpoint  | verify `/health` and `/api/inference/predict` |
| open the frontend demo     | show the browser-based prototype              |

The supervisor checklist explains the demo flow:

```text
frontend → FastAPI backend → API wrapper → inference pipeline → hierarchical decision → frontend result
```

Expected demo result for `local_000001`:

| Field                  | Value                |
| ---------------------- | -------------------- |
| pred_label             | plastic              |
| max_softmax_confidence | 0.962933             |
| decision_type          | fine_label           |
| final_label            | plastic              |
| reason                 | fine_confidence_high |

Research note:

This stage improves project presentation and reproducibility. It gives a clear guide for running the prototype and a structured checklist for explaining the project novelty during a supervisor meeting or final viva.

## Final Prototype Summary Report v1 Summary

This stage creates the final prototype summary documents for the current OpenWaste-HR implementation.

Created files:

* docs/results/final_prototype_summary_v1.md
* docs/supervisor_updates/final_prototype_summary_v1.md
* tests/test_final_prototype_summary_docs.py

The final prototype summary documents the full workflow:

```text
dataset preparation → baseline training → reject option → hierarchical decision policy → safe policy tuning → local unknown evaluation → active learning → inference → backend → frontend demo
```

Key current results:

| Area                                               | Result     |
| -------------------------------------------------- | ---------- |
| Closed-set baseline test accuracy                  | 0.692708   |
| Confidence reject local unknown rejection rate     | 0.350000   |
| Safe hierarchical local unknown manual-review rate | 0.375000   |
| Active learning candidates selected                | 20         |
| Human labelling sheet rows                         | 20         |
| Reviewed rows currently ready for dataset          | 0          |
| Single-image demo final label                      | plastic    |
| Single-image demo decision type                    | fine_label |
| Frontend/backend demo status                       | working    |

Current implemented prototype interfaces:

| Interface                           | Status  |
| ----------------------------------- | ------- |
| command-line single-image inference | working |
| command-line batch inference        | working |
| prototype API wrapper               | working |
| FastAPI backend endpoint            | working |
| simple frontend demo                | working |

Research note:

This stage consolidates the project into a thesis-ready implementation summary. It clearly presents OpenWaste-HR as a hierarchical open-set waste classification prototype with reject/manual-review decisions and local active learning support.

## Thesis Implementation Chapter Draft v1 Summary

This stage creates a thesis-ready implementation chapter draft and a supervisor summary.

Created files:

* docs/thesis/implementation_chapter_draft_v1.md
* docs/supervisor_updates/implementation_chapter_summary_v1.md
* tests/test_implementation_chapter_docs.py

The implementation chapter draft covers:

| Section                      | Purpose                                                               |
| ---------------------------- | --------------------------------------------------------------------- |
| Introduction                 | explains OpenWaste-HR as a hierarchical open-set prototype            |
| System overview              | summarises the full project workflow                                  |
| Taxonomy design              | explains fine labels, coarse labels, unknown, and manual_review       |
| Dataset preparation          | describes manifest building, splits, and inspection                   |
| Baseline model               | describes the MobileNetV3-style classifier                            |
| Reject-option baselines      | explains confidence, max-logit, and energy scoring                    |
| Hierarchical decision policy | explains fine_label, coarse_label, and manual_review outputs          |
| Safe policy tuning           | explains stricter thresholds and the safety-coverage trade-off        |
| Active learning workflow     | explains candidate selection and human labelling                      |
| Inference pipeline           | explains single-image and batch inference                             |
| API wrapper                  | explains the backend-friendly response format                         |
| Backend                      | explains the FastAPI `/health` and `/api/inference/predict` endpoints |
| Frontend                     | explains the simple browser demo                                      |
| Testing                      | summarises automated test coverage                                    |
| Limitations                  | identifies current prototype limitations                              |
| Conclusion                   | summarises the implemented contribution                               |

Key thesis message:

```text
OpenWaste-HR should be presented as hierarchical open-set waste classification with reject/manual-review decisions and local active learning support, not only as a normal CNN classifier.
```

Research note:

This stage converts the implemented prototype into thesis-ready writing. It will be used as the base for the final implementation chapter, with diagrams and screenshots added later.

## Architecture Diagram Source File v1 Summary

This stage creates a Mermaid architecture diagram source file and explanation notes for the current OpenWaste-HR prototype.

Created files:

* docs/architecture/openwaste_hr_architecture_v1.mmd
* docs/architecture/openwaste_hr_architecture_notes_v1.md
* tests/test_architecture_diagram_docs.py

The architecture diagram covers the following layers:

| Layer                         | Purpose                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------- |
| Dataset and Data Management   | known data, local unknown data, manifests, splits, inspection                            |
| Model Training and Evaluation | baseline training, closed-set evaluation, reject baselines, local unknown evaluation     |
| Hierarchical Decision Layer   | taxonomy, fine_label, coarse_label, manual_review, safe policy tuning                    |
| Local Active Learning Loop    | candidate selection, human labelling sheet, reviewed label processing, future dataset v2 |
| Inference Layer               | single-image inference, batch inference, prototype API wrapper                           |
| Backend Layer                 | FastAPI backend, `/health`, `/api/inference/predict`                                     |
| Frontend Demo                 | browser UI, image path input, final decision display                                     |

Research note:

This stage prepares a thesis-ready architecture diagram source. The Mermaid file can later be exported as PNG or SVG and included in the implementation chapter.

## Thesis Evaluation Chapter Draft v1 Summary

This stage creates the first thesis-ready evaluation chapter draft and supervisor evaluation summary.

Created files:

* docs/thesis/evaluation_chapter_draft_v1.md
* docs/supervisor_updates/evaluation_chapter_summary_v1.md
* tests/test_evaluation_chapter_docs.py

The evaluation chapter draft covers:

| Section                        | Purpose                                                                  |
| ------------------------------ | ------------------------------------------------------------------------ |
| Evaluation objectives          | explains what the current prototype evaluation measures                  |
| Experimental setup             | describes the prototype evaluation workflow                              |
| Dataset and splits             | summarises known data and local unknown evaluation data                  |
| Evaluation metrics             | explains closed-set, reject-option, unknown, and hierarchical metrics    |
| Closed-set baseline            | reports forced-classification baseline performance                       |
| Reject-option baselines        | reports confidence, max-logit, and energy behaviour                      |
| Local unknown evaluation       | evaluates unknown rejection and false acceptance                         |
| Hierarchical policy evaluation | compares fine_label, coarse_label, and manual_review behaviour           |
| Safe policy tuning             | explains the safety-coverage trade-off                                   |
| Policy comparison              | compares closed-set, reject, hierarchical, and safe hierarchical systems |
| Active learning evaluation     | summarises candidate selection and pending human review                  |
| Inference/prototype validation | confirms command-line, backend, and frontend workflows                   |
| Limitations                    | explains current limitations                                             |
| Future evaluation plan         | records pretrained, extra dataset, and active-learning v2 plans          |

Key current results:

| Result                                             |      Value |
| -------------------------------------------------- | ---------: |
| Closed-set baseline accuracy                       |   0.692708 |
| Confidence reject selective accuracy               |   0.770992 |
| Confidence reject local unknown rejection rate     |   0.350000 |
| Safe hierarchical accepted reliability             |   0.889328 |
| Safe hierarchical local unknown manual-review rate |   0.375000 |
| Active learning candidates selected                |         20 |
| Current demo prediction                            |    plastic |
| Current demo decision type                         | fine_label |

Research note:

This chapter is a v1 evaluation draft for the current working prototype. It is not the final model-comparison chapter. Later, after pretrained training, additional datasets, human label correction, and retraining, the evaluation chapter will be updated with final comparison results.

## Thesis Methodology Chapter Consolidation v1 Summary

This stage creates a consolidated thesis methodology chapter draft and supervisor methodology summary.

Created files:

* docs/thesis/methodology_chapter_consolidated_v1.md
* docs/supervisor_updates/methodology_chapter_summary_v1.md
* tests/test_methodology_chapter_docs.py

The methodology chapter consolidates the project workflow:

```text id="ip9qn5"
taxonomy design → dataset preparation → baseline model training → closed-set evaluation → reject-option evaluation → local unknown evaluation → hierarchical decision policy → safe policy tuning → active learning workflow → inference pipeline → backend/frontend prototype
```

Key methodology areas covered:

| Area                              | Purpose                                                            |
| --------------------------------- | ------------------------------------------------------------------ |
| Taxonomy methodology              | defines fine labels, coarse labels, unknown, and manual_review     |
| Dataset methodology               | uses manifest-based dataset preparation                            |
| Dataset inspection                | checks image validity, labels, sizes, and imbalance                |
| Baseline training                 | trains the first closed-set classifier                             |
| Reject-option methodology         | evaluates confidence, max-logit, and energy rejection              |
| Local unknown evaluation          | measures unknown rejection and false acceptance                    |
| Hierarchical decision methodology | supports fine_label, coarse_label, and manual_review               |
| Safe policy tuning                | selects stricter thresholds for safer decisions                    |
| Active learning methodology       | selects local candidates for human review                          |
| Human labelling methodology       | prepares reviewed local data for future dataset updates            |
| Inference methodology             | supports single-image, batch, API, backend, and frontend workflows |
| Testing methodology               | supports reproducibility and implementation confidence             |

Research note:

This stage converts the separate methodology documents into a thesis-ready methodology chapter draft. It clearly presents OpenWaste-HR as a hierarchical open-set waste classification methodology rather than only a normal CNN classification workflow.

## Pretrained Training Preparation v1 Summary

This stage prepares the pretrained transfer-learning model configuration and documentation.

Created files:

* ml/configs/train_pretrained_trashnet.yaml
* docs/methodology/pretrained_training_plan_v1.md
* docs/supervisor_updates/pretrained_training_plan_summary_v1.md
* tests/test_pretrained_training_preparation.py

The current model is treated as:

| Model      | Description                             |
| ---------- | --------------------------------------- |
| Baseline A | scratch-trained TrashNet-style baseline |

The next model is prepared as:

| Model      | Description                           |
| ---------- | ------------------------------------- |
| Baseline B | pretrained transfer-learning baseline |

The pretrained config enables:

```yaml id="o1pbqe"
pretrained: true
```

The output naming is separated using:

```text id="28ir21"
pretrained_trashnet_v1
```

Planned comparison:

| Comparison                  | Purpose                                                                   |
| --------------------------- | ------------------------------------------------------------------------- |
| Baseline A vs Baseline B    | compare scratch training with pretrained transfer learning                |
| Closed-set metrics          | compare known-test accuracy, balanced accuracy, macro-F1, and weighted-F1 |
| Reject-option metrics       | compare selective accuracy and rejection behaviour                        |
| Local unknown metrics       | compare manual-review/rejection rate and false acceptance                 |
| Hierarchical policy metrics | compare fine_label, coarse_label, and manual_review decisions             |

Research note:

This stage prepares the next model-improvement phase. The pretrained model will not replace the OpenWaste-HR contribution; it will test whether stronger image features improve the hierarchical open-set decision workflow.

## Pretrained Training Smoke Test v1 Summary

This stage ran a smoke test for the pretrained transfer-learning training setup.

Created files:

* ml/configs/train_pretrained_trashnet_smoke.yaml
* docs/methodology/pretrained_training_smoke_test_v1.md
* docs/supervisor_updates/pretrained_training_smoke_test_summary_v1.md
* tests/test_pretrained_training_smoke_test_docs.py

Test result:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   154 |
| Warnings          |     1 |
| Blocking failures |     0 |

The pretrained smoke training successfully downloaded and loaded pretrained model weights.

Smoke training result:

| Metric                              |  Value |
| ----------------------------------- | -----: |
| Device                              |   cuda |
| Epochs                              |      1 |
| Best epoch                          |      1 |
| Training loss                       | 2.0271 |
| Training accuracy                   | 0.5674 |
| Validation loss                     | 1.2653 |
| Validation accuracy                 | 0.6817 |
| Validation macro-F1                 | 0.6300 |
| Test accuracy using best checkpoint | 0.7083 |
| Test macro-F1 using best checkpoint | 0.6077 |

Generated smoke-test outputs:

| Output               | Path                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------- |
| Training metrics CSV | ml/outputs/metrics/pretrained_trashnet_smoke_v1_training_metrics.csv                      |
| Test metrics JSON    | ml/outputs/metrics/pretrained_trashnet_smoke_v1_test_metrics.json                         |
| Best checkpoint      | ml/outputs/checkpoints/pretrained_trashnet_smoke_v1/pretrained_trashnet_smoke_v1_best.pt  |
| Final checkpoint     | ml/outputs/checkpoints/pretrained_trashnet_smoke_v1/pretrained_trashnet_smoke_v1_final.pt |

Research note:

The smoke test confirms that the pretrained transfer-learning setup can run successfully with `pretrained: true`, load pretrained weights, train for one epoch, evaluate on the test set, and save outputs separately from the original scratch-trained baseline. This result is not used as the final pretrained comparison because it was only a one-epoch smoke test. The next stage is full pretrained training.

## Full Pretrained Model Training v1 Summary

This stage trained the full pretrained transfer-learning model for OpenWaste-HR.

Created files:

* docs/methodology/pretrained_training_v1.md
* docs/supervisor_updates/pretrained_training_summary_v1.md
* tests/test_pretrained_training_docs.py

Test result before training:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   159 |
| Warnings          |     1 |
| Blocking failures |     0 |

The full pretrained training used:

| Setting                 | Value                                            |
| ----------------------- | ------------------------------------------------ |
| Config                  | ml/configs/train_pretrained_trashnet.yaml        |
| Device                  | cuda                                             |
| Classes                 | paper_cardboard, plastic, glass, metal, residual |
| Epoch limit             | 50                                               |
| Early stopping patience | 7                                                |
| Early stopping monitor  | val_macro_f1                                     |

Training result:

| Metric                              |  Value |
| ----------------------------------- | -----: |
| Best epoch                          |     19 |
| Best validation macro-F1            | 0.8577 |
| Early stopping epoch                |     26 |
| Test accuracy using best checkpoint | 0.8880 |
| Test macro-F1 using best checkpoint | 0.8510 |

Generated pretrained outputs:

| Output               | Path                                                                          |
| -------------------- | ----------------------------------------------------------------------------- |
| Training metrics CSV | ml/outputs/metrics/pretrained_trashnet_v1_training_metrics.csv                |
| Test metrics JSON    | ml/outputs/metrics/pretrained_trashnet_v1_test_metrics.json                   |
| Best checkpoint      | ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt  |
| Final checkpoint     | ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_final.pt |

Initial comparison with the scratch-trained baseline:

| Model                                            | Test Accuracy | Test Macro-F1 |
| ------------------------------------------------ | ------------: | ------------: |
| Baseline A: scratch-trained TrashNet-style model |        0.6927 |        0.6456 |
| Baseline B: pretrained transfer-learning model   |        0.8880 |        0.8510 |
| Improvement                                      |       +0.1953 |       +0.2054 |

Research note:

The pretrained transfer-learning model substantially improves known-class classification performance compared with the scratch-trained baseline. This confirms that pretrained visual features are valuable for the OpenWaste-HR pipeline. The next stage is to evaluate this pretrained checkpoint through the same downstream workflow: closed-set evaluation, reject-option evaluation, local unknown evaluation, hierarchical decision policy, and safe policy tuning.

## Pretrained Baseline Evaluation v1 Summary

This stage evaluated the full pretrained transfer-learning model using the formal closed-set evaluation pipeline.

Created files:

* ml/configs/evaluate_pretrained_trashnet.yaml
* docs/methodology/pretrained_baseline_evaluation_v1.md
* docs/supervisor_updates/pretrained_baseline_evaluation_summary_v1.md
* tests/test_pretrained_baseline_evaluation_docs.py
* docs/results/pretrained_trashnet_v1_report.md

Test result before evaluation:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   164 |
| Warnings          |     1 |
| Blocking failures |     0 |

Pretrained baseline evaluation result:

| Metric            |  Value |
| ----------------- | -----: |
| Test samples      |    384 |
| Accuracy          | 0.8880 |
| Balanced accuracy | 0.8431 |
| Macro-F1          | 0.8510 |
| Weighted-F1       | 0.8873 |

Generated evaluation outputs:

| Output                | Path                                                                |
| --------------------- | ------------------------------------------------------------------- |
| Predictions CSV       | ml/outputs/metrics/pretrained_trashnet_test_predictions_v1.csv      |
| Metrics JSON          | ml/outputs/metrics/pretrained_trashnet_eval_metrics_v1.json         |
| Confusion matrix CSV  | ml/outputs/metrics/pretrained_trashnet_confusion_matrix_v1.csv      |
| Classification report | ml/outputs/metrics/pretrained_trashnet_classification_report_v1.csv |
| Confusion matrix plot | ml/outputs/figures/pretrained_trashnet_confusion_matrix_v1.png      |
| Thesis report         | docs/results/pretrained_trashnet_v1_report.md                       |

Comparison with scratch-trained baseline:

| Model                                            | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ------------------------------------------------ | -------: | ----------------: | -------: | ----------: |
| Baseline A: scratch-trained TrashNet-style model |   0.6927 |            0.6545 |   0.6456 |      0.7009 |
| Baseline B: pretrained transfer-learning model   |   0.8880 |            0.8431 |   0.8510 |      0.8873 |
| Improvement                                      |  +0.1953 |           +0.1886 |  +0.2054 |     +0.1864 |

Research note:

The formal pretrained evaluation confirms that Baseline B substantially improves known-class classification compared with Baseline A. The next stage is to test whether this stronger checkpoint also improves reject-option behaviour, local unknown handling, and hierarchical decision safety.

## Pretrained Reject-Option Evaluation v1 Summary

This stage evaluated reject-option methods using the full pretrained transfer-learning checkpoint.

Created files:

* ml/configs/confidence_reject_pretrained_trashnet.yaml
* ml/configs/open_set_score_pretrained_trashnet.yaml
* docs/methodology/pretrained_reject_option_evaluation_v1.md
* docs/supervisor_updates/pretrained_reject_option_evaluation_summary_v1.md
* tests/test_pretrained_reject_option_evaluation_docs.py
* docs/results/pretrained_confidence_reject_baseline_v1_report.md
* docs/results/pretrained_open_set_score_baseline_v1_report.md

Test result before evaluation:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   169 |
| Warnings          |     1 |
| Blocking failures |     0 |

Checkpoint used:

| Field                 | Value                                                                        |
| --------------------- | ---------------------------------------------------------------------------- |
| Model                 | Baseline B: pretrained transfer-learning model                               |
| Checkpoint            | ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt |
| Best checkpoint epoch | 19                                                                           |
| Device                | cuda                                                                         |

### Confidence-Threshold Reject Result

Selected threshold:

| Metric    |  Value |
| --------- | -----: |
| Threshold | 0.9900 |

Validation result:

| Metric                |    Value |
| --------------------- | -------: |
| Total samples         |      377 |
| Accepted count        |      269 |
| Rejected count        |      108 |
| Coverage              | 0.713528 |
| Rejection rate        | 0.286472 |
| Forced accuracy       | 0.883289 |
| Selective accuracy    | 0.973978 |
| Selective error rate  | 0.026022 |
| Selective macro-F1    | 0.958167 |
| Selective weighted-F1 | 0.974375 |

Test result:

| Metric                     |    Value |
| -------------------------- | -------: |
| Total samples              |      384 |
| Accepted count             |      277 |
| Rejected count             |      107 |
| Coverage                   | 0.721354 |
| Rejection rate             | 0.278646 |
| Selective error rate       | 0.046931 |
| Derived selective accuracy | 0.953069 |
| Selective macro-F1         | 0.923975 |
| Selective weighted-F1      | 0.952995 |

Generated confidence-reject outputs:

| Output                 | Path                                                                        |
| ---------------------- | --------------------------------------------------------------------------- |
| Validation predictions | ml/outputs/metrics/pretrained_confidence_reject_val_predictions_v1.csv      |
| Test predictions       | ml/outputs/metrics/pretrained_confidence_reject_test_predictions_v1.csv     |
| Threshold sweep        | ml/outputs/metrics/pretrained_confidence_reject_threshold_sweep_v1.csv      |
| Selected threshold     | ml/outputs/metrics/pretrained_confidence_reject_selected_threshold_v1.json  |
| Test metrics           | ml/outputs/metrics/pretrained_confidence_reject_test_metrics_v1.json        |
| Coverage-risk plot     | ml/outputs/figures/pretrained_confidence_reject_coverage_risk_v1.png        |
| Confidence histogram   | ml/outputs/figures/pretrained_confidence_reject_confidence_histogram_v1.png |
| Thesis report          | docs/results/pretrained_confidence_reject_baseline_v1_report.md             |

### Max-Logit and Energy Reject Result

Selected thresholds:

| Method    | Threshold | Accept Direction |
| --------- | --------: | ---------------- |
| Max-logit |  8.021357 | greater_equal    |
| Energy    | -7.885723 | less_equal       |

Max-logit validation result:

| Metric                |    Value |
| --------------------- | -------: |
| Total samples         |      377 |
| Accepted count        |      265 |
| Rejected count        |      112 |
| Coverage              | 0.702918 |
| Rejection rate        | 0.297082 |
| Forced accuracy       | 0.883289 |
| Selective accuracy    | 0.958491 |
| Selective error rate  | 0.041509 |
| Selective macro-F1    | 0.948101 |
| Selective weighted-F1 | 0.958383 |

Energy validation result:

| Metric                |    Value |
| --------------------- | -------: |
| Total samples         |      377 |
| Accepted count        |      272 |
| Rejected count        |      105 |
| Coverage              | 0.721485 |
| Rejection rate        | 0.278515 |
| Forced accuracy       | 0.883289 |
| Selective accuracy    | 0.952206 |
| Selective error rate  | 0.047794 |
| Selective macro-F1    | 0.941689 |
| Selective weighted-F1 | 0.951976 |

Max-logit test result:

| Metric                |    Value |
| --------------------- | -------: |
| Total samples         |      384 |
| Accepted count        |      282 |
| Rejected count        |      102 |
| Coverage              | 0.734375 |
| Rejection rate        | 0.265625 |
| Forced accuracy       | 0.888021 |
| Selective accuracy    | 0.957447 |
| Selective error rate  | 0.042553 |
| Selective macro-F1    | 0.942696 |
| Selective weighted-F1 | 0.957535 |

Energy test result:

| Metric                |    Value |
| --------------------- | -------: |
| Total samples         |      384 |
| Accepted count        |      286 |
| Rejected count        |       98 |
| Coverage              | 0.744792 |
| Rejection rate        | 0.255208 |
| Forced accuracy       | 0.888021 |
| Selective accuracy    | 0.958042 |
| Selective error rate  | 0.041958 |
| Selective macro-F1    | 0.943450 |
| Selective weighted-F1 | 0.958130 |

Generated open-set score outputs:

| Output              | Path                                                                     |
| ------------------- | ------------------------------------------------------------------------ |
| Validation scores   | ml/outputs/metrics/pretrained_open_set_score_val_predictions_v1.csv      |
| Test scores         | ml/outputs/metrics/pretrained_open_set_score_test_predictions_v1.csv     |
| Threshold sweep     | ml/outputs/metrics/pretrained_open_set_score_threshold_sweep_v1.csv      |
| Selected thresholds | ml/outputs/metrics/pretrained_open_set_score_selected_thresholds_v1.json |
| Test metrics        | ml/outputs/metrics/pretrained_open_set_score_test_metrics_v1.json        |
| Coverage-risk plot  | ml/outputs/figures/pretrained_open_set_score_coverage_risk_v1.png        |
| Score histogram     | ml/outputs/figures/pretrained_open_set_score_histogram_v1.png            |
| Thesis report       | docs/results/pretrained_open_set_score_baseline_v1_report.md             |

### Comparison with Scratch-Trained Reject Baselines

| Method               | Model               | Coverage | Selective Accuracy | Selective Macro-F1 |
| -------------------- | ------------------- | -------: | -----------------: | -----------------: |
| Confidence threshold | Scratch baseline    | 0.682292 |           0.770992 |           0.715164 |
| Confidence threshold | Pretrained baseline | 0.721354 |   0.953069 derived |           0.923975 |
| Max-logit            | Scratch baseline    | 0.716146 |           0.760000 |           0.723257 |
| Max-logit            | Pretrained baseline | 0.734375 |           0.957447 |           0.942696 |
| Energy               | Scratch baseline    | 0.744792 |           0.741259 |           0.705525 |
| Energy               | Pretrained baseline | 0.744792 |           0.958042 |           0.943450 |

Research note:

The pretrained checkpoint substantially improves reject-option behaviour on the known test set. Compared with the scratch-trained baseline, the pretrained model achieves higher coverage and much higher accepted-decision reliability. The next stage is to evaluate whether this improvement also helps on the local unknown dataset, where the main concern is unknown false acceptance.

## Pretrained Local Unknown Evaluation v1 Summary

This stage evaluated the full pretrained transfer-learning checkpoint on the local unknown dataset.

Created files:

* ml/configs/pretrained_local_unknown_evaluation.yaml
* docs/methodology/pretrained_local_unknown_evaluation_v1.md
* docs/supervisor_updates/pretrained_local_unknown_evaluation_summary_v1.md
* tests/test_pretrained_local_unknown_evaluation_docs.py
* docs/results/pretrained_local_unknown_evaluation_v1_report.md

Test result before evaluation:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   174 |
| Warnings          |     1 |
| Blocking failures |     0 |

Checkpoint used:

| Field                 | Value                                                                        |
| --------------------- | ---------------------------------------------------------------------------- |
| Model                 | Baseline B: pretrained transfer-learning model                               |
| Checkpoint            | ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt |
| Best checkpoint epoch | 19                                                                           |
| Device                | cuda                                                                         |

Pretrained local unknown evaluation result:

| Method               | Total Unknown Samples | Rejected Unknown Count | Accepted Unknown as Known Count | Unknown Rejection Rate | Unknown False Acceptance Rate |
| -------------------- | --------------------: | ---------------------: | ------------------------------: | ---------------------: | ----------------------------: |
| Confidence threshold |                    40 |                     20 |                              20 |               0.500000 |                      0.500000 |
| Max-logit score      |                    40 |                     10 |                              30 |               0.250000 |                      0.750000 |
| Energy score         |                    40 |                     10 |                              30 |               0.250000 |                      0.750000 |

Generated outputs:

| Output                          | Path                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------ |
| Predictions CSV                 | ml/outputs/metrics/pretrained_local_unknown_predictions_v1.csv                 |
| Method decisions CSV            | ml/outputs/metrics/pretrained_local_unknown_method_decisions_v1.csv            |
| Metrics JSON                    | ml/outputs/metrics/pretrained_local_unknown_evaluation_metrics_v1.json         |
| Accepted label distribution CSV | ml/outputs/metrics/pretrained_local_unknown_accepted_label_distribution_v1.csv |
| Rejection-rate plot             | ml/outputs/figures/pretrained_local_unknown_rejection_rates_v1.png             |
| Accepted-label plot             | ml/outputs/figures/pretrained_local_unknown_accepted_label_distribution_v1.png |
| Thesis report                   | docs/results/pretrained_local_unknown_evaluation_v1_report.md                  |

Comparison with scratch-trained local unknown evaluation:

| Method               | Scratch Unknown Rejection Rate | Pretrained Unknown Rejection Rate | Scratch False Acceptance Rate | Pretrained False Acceptance Rate |
| -------------------- | -----------------------------: | --------------------------------: | ----------------------------: | -------------------------------: |
| Confidence threshold |                       0.350000 |                          0.500000 |                      0.650000 |                         0.500000 |
| Max-logit score      |                       0.275000 |                          0.250000 |                      0.725000 |                         0.750000 |
| Energy score         |                       0.200000 |                          0.250000 |                      0.800000 |                         0.750000 |

Research note:

The pretrained model substantially improves known-class classification and reject-option reliability on the known test set. For local unknown handling, the confidence threshold improves meaningfully from 0.350000 to 0.500000 unknown rejection rate. However, max-logit and energy do not show the same level of improvement. This confirms that stronger closed-set performance does not automatically solve unknown handling. The next stage should evaluate the pretrained checkpoint using hierarchical decision policy and safe policy tuning.

## Pretrained Hierarchical Policy Evaluation v1 Summary

This stage evaluated the hierarchical decision policy using the full pretrained transfer-learning checkpoint.

Created files:

* ml/configs/pretrained_hierarchical_decision_policy.yaml
* docs/methodology/pretrained_hierarchical_policy_evaluation_v1.md
* docs/supervisor_updates/pretrained_hierarchical_policy_evaluation_summary_v1.md
* tests/test_pretrained_hierarchical_policy_evaluation_docs.py
* docs/results/pretrained_hierarchical_decision_policy_v1_report.md

Test result before evaluation:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   179 |
| Warnings          |     1 |
| Blocking failures |     0 |

Policy used:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.990000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.350000 |

Known-test result:

| Metric                                  |    Value |
| --------------------------------------- | -------: |
| Known total samples                     |      384 |
| Fine decisions                          |      277 |
| Coarse fallback decisions               |       98 |
| Manual-review decisions                 |        9 |
| Known decision coverage                 | 0.976562 |
| Known manual-review rate                | 0.023438 |
| Fine correct count                      |      264 |
| Coarse correct count                    |       95 |
| Hierarchical success count              |      359 |
| Fine accuracy on fine decisions         | 0.953069 |
| Coarse accuracy on coarse decisions     | 0.969388 |
| Hierarchical success rate over all      | 0.934896 |
| Hierarchical success rate over accepted | 0.957333 |

Local unknown result:

| Metric                      |    Value |
| --------------------------- | -------: |
| Unknown total samples       |       40 |
| Unknown manual-review count |        3 |
| Unknown fine accept count   |        6 |
| Unknown coarse accept count |       31 |
| Unknown accepted count      |       37 |
| Unknown manual-review rate  | 0.075000 |
| Unknown acceptance rate     | 0.925000 |

Generated outputs:

| Output                      | Path                                                                       |
| --------------------------- | -------------------------------------------------------------------------- |
| Known decisions CSV         | ml/outputs/metrics/pretrained_hierarchical_known_test_decisions_v1.csv     |
| Local unknown decisions CSV | ml/outputs/metrics/pretrained_hierarchical_local_unknown_decisions_v1.csv  |
| Metrics JSON                | ml/outputs/metrics/pretrained_hierarchical_decision_policy_metrics_v1.json |
| Decision distribution CSV   | ml/outputs/metrics/pretrained_hierarchical_decision_distribution_v1.csv    |
| Decision plot               | ml/outputs/figures/pretrained_hierarchical_decision_distribution_v1.png    |
| Thesis report               | docs/results/pretrained_hierarchical_decision_policy_v1_report.md          |

Comparison note:

The pretrained hierarchical policy greatly improves known-test decision quality compared with the scratch-trained hierarchical policy. Known-test coverage increased to 0.976562 and accepted hierarchical reliability increased to 0.957333. However, local unknown manual-review rate remained low at 0.075000 because many local unknown images were accepted as coarse fallback decisions.

Research note:

This confirms that pretrained features improve known-class and hierarchical known-test performance, but a strong classifier can still accept local unknown images too easily. The next stage should tune a safer pretrained hierarchical policy to improve the local unknown manual-review rate while preserving useful known-test coverage.

## Safe Pretrained Hierarchical Policy Tuning v1 Summary

This stage tuned a safer hierarchical decision policy for the full pretrained transfer-learning checkpoint.

Created files:

* ml/configs/pretrained_safe_hierarchical_policy_tuning.yaml
* docs/methodology/pretrained_safe_hierarchical_policy_tuning_v1.md
* docs/supervisor_updates/pretrained_safe_hierarchical_policy_tuning_summary_v1.md
* tests/test_pretrained_safe_hierarchical_policy_tuning_docs.py
* docs/results/pretrained_safe_hierarchical_policy_tuning_v1_report.md

Test result before tuning:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   184 |
| Warnings          |     1 |
| Blocking failures |     0 |

Tuning result:

| Metric                         | Value |
| ------------------------------ | ----: |
| Threshold candidates evaluated |   750 |

Selected policy:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

Known-test result:

| Metric                                  |    Value |
| --------------------------------------- | -------: |
| Known total samples                     |      384 |
| Fine decisions                          |      262 |
| Coarse fallback decisions               |       70 |
| Manual-review decisions                 |       52 |
| Known decision coverage                 | 0.864583 |
| Known manual-review rate                | 0.135417 |
| Fine correct count                      |      251 |
| Coarse correct count                    |       68 |
| Hierarchical success count              |      319 |
| Fine accuracy on fine decisions         | 0.958015 |
| Coarse accuracy on coarse decisions     | 0.971429 |
| Hierarchical success rate over all      | 0.830729 |
| Hierarchical success rate over accepted | 0.960843 |

Local unknown result:

| Metric                      |    Value |
| --------------------------- | -------: |
| Unknown total samples       |       40 |
| Unknown manual-review count |       24 |
| Unknown fine accept count   |        6 |
| Unknown coarse accept count |       10 |
| Unknown accepted count      |       16 |
| Unknown manual-review rate  | 0.600000 |
| Unknown acceptance rate     | 0.400000 |

Generated outputs:

| Output                      | Path                                                                           |
| --------------------------- | ------------------------------------------------------------------------------ |
| Threshold sweep             | ml/outputs/metrics/pretrained_safe_hierarchical_threshold_sweep_v1.csv         |
| Selected policy             | ml/outputs/metrics/pretrained_safe_hierarchical_selected_policy_v1.json        |
| Known decisions CSV         | ml/outputs/metrics/pretrained_safe_hierarchical_known_test_decisions_v1.csv    |
| Local unknown decisions CSV | ml/outputs/metrics/pretrained_safe_hierarchical_local_unknown_decisions_v1.csv |
| Metrics JSON                | ml/outputs/metrics/pretrained_safe_hierarchical_policy_metrics_v1.json         |
| Decision distribution CSV   | ml/outputs/metrics/pretrained_safe_hierarchical_decision_distribution_v1.csv   |
| Tuning plot                 | ml/outputs/figures/pretrained_safe_hierarchical_policy_tuning_v1.png           |
| Thesis report               | docs/results/pretrained_safe_hierarchical_policy_tuning_v1_report.md           |

Comparison with earlier policies:

| Policy                              | Known Coverage | Accepted Reliability | Local Unknown Manual Review Rate | Local Unknown Acceptance Rate |
| ----------------------------------- | -------------: | -------------------: | -------------------------------: | ----------------------------: |
| Scratch safe hierarchical policy    |       0.658854 |             0.889328 |                         0.375000 |                      0.625000 |
| Pretrained hierarchical policy v1   |       0.976562 |             0.957333 |                         0.075000 |                      0.925000 |
| Pretrained safe hierarchical policy |       0.864583 |             0.960843 |                         0.600000 |                      0.400000 |

Research note:

The safe pretrained hierarchical policy is the best current OpenWaste-HR decision policy. Compared with the scratch safe hierarchical policy, it improves known-test coverage, accepted-decision reliability, and local unknown manual-review rate. Compared with the first pretrained hierarchical policy, it greatly improves local unknown handling while preserving strong known-test reliability. This policy should currently be treated as the best candidate for the final OpenWaste-HR prototype.

## Final Model and Policy Comparison Table v1 Summary

This stage creates the first complete model and policy comparison table after pretrained training and safe pretrained hierarchical tuning.

Created files:

* docs/results/final_model_policy_comparison_v1.md
* docs/supervisor_updates/final_model_policy_comparison_summary_v1.md
* tests/test_final_model_policy_comparison_docs.py

Main closed-set comparison:

| Model                             |  Accuracy | Balanced Accuracy |  Macro-F1 | Weighted-F1 |
| --------------------------------- | --------: | ----------------: | --------: | ----------: |
| Baseline A: scratch-trained model |  0.692708 |          0.654500 |  0.645600 |    0.700900 |
| Baseline B: pretrained model      |  0.888000 |          0.843100 |  0.851000 |    0.887300 |
| Improvement                       | +0.195292 |         +0.188600 | +0.205400 |   +0.186400 |

Current best OpenWaste-HR policy:

| Item                              | Value                               |
| --------------------------------- | ----------------------------------- |
| Best current policy               | Pretrained Safe Hierarchical Policy |
| Known-test coverage               | 0.864583                            |
| Accepted hierarchical reliability | 0.960843                            |
| Local unknown manual-review rate  | 0.600000                            |
| Local unknown acceptance rate     | 0.400000                            |

Comparison with earlier safe policy:

| Metric                           | Scratch Safe Hierarchical | Pretrained Safe Hierarchical |
| -------------------------------- | ------------------------: | ---------------------------: |
| Known coverage                   |                  0.658854 |                     0.864583 |
| Accepted reliability             |                  0.889328 |                     0.960843 |
| Local unknown manual-review rate |                  0.375000 |                     0.600000 |
| Local unknown acceptance rate    |                  0.625000 |                     0.400000 |

Research note:

The pretrained safe hierarchical policy is the best current OpenWaste-HR system. It improves known-test performance, accepted-decision reliability, and local unknown manual-review rate compared with the scratch-trained safe policy. This result supports the project argument that OpenWaste-HR should be presented as hierarchical open-set waste classification with reject/manual-review decisions and local active learning support, not only as a normal waste classifier.

## Best Pretrained Safe Policy Integration v1 Summary

This stage updated the active OpenWaste-HR prototype to use the best current model and decision policy.

Active system:

| Component       | Selected Version                                                             |
| --------------- | ---------------------------------------------------------------------------- |
| Model           | Baseline B: pretrained transfer-learning model                               |
| Checkpoint      | ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt |
| Decision policy | pretrained safe hierarchical policy                                          |

Active thresholds:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

Files created or updated:

* docs/methodology/best_pretrained_safe_policy_integration_v1.md
* docs/supervisor_updates/best_pretrained_safe_policy_integration_summary_v1.md
* tests/test_best_pretrained_safe_policy_integration.py
* ml/configs/inference_pipeline.yaml
* ml/configs/batch_inference_pipeline.yaml
* ml/configs/prototype_api_wrapper.yaml
* frontend/index.html

Test result:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   196 |
| Warnings          |     1 |
| Blocking failures |     0 |

Single-image inference test:

| Field                  | Value                                         |
| ---------------------- | --------------------------------------------- |
| Sample ID              | local_000001                                  |
| Image                  | ml/data/local_unknown/images/local_000001.jpg |
| Device                 | cuda                                          |
| Predicted fine label   | paper_cardboard                               |
| Max softmax confidence | 0.993654                                      |
| Top coarse label       | recyclable                                    |
| Top coarse confidence  | 0.999999                                      |
| Decision type          | coarse_label                                  |
| Final label            | recyclable                                    |
| Final confidence       | 0.999999                                      |
| Decision reason        | coarse_fallback_stable                        |

API wrapper test:

| Field                | Value                |
| -------------------- | -------------------- |
| Request ID           | best_policy_demo_001 |
| Status               | success              |
| Predicted fine label | paper_cardboard      |
| Final decision type  | coarse_label         |
| Final label          | recyclable           |
| Final confidence     | 0.999999             |

Generated outputs:

| Output                       | Path                                                                          |
| ---------------------------- | ----------------------------------------------------------------------------- |
| Single-image JSON result     | ml/outputs/metrics/best_pretrained_safe_single_image_inference_result_v1.json |
| Single-image Markdown result | ml/outputs/metrics/best_pretrained_safe_single_image_inference_result_v1.md   |
| API JSON response            | ml/outputs/metrics/best_pretrained_safe_prototype_api_response_v1.json        |
| API Markdown response        | ml/outputs/metrics/best_pretrained_safe_prototype_api_response_v1.md          |

Research note:

The local_000001 image is actually a rubber slipper / flip-flop, which is not part of the known TrashNet-style training labels. The model predicted paper_cardboard with high softmax confidence and returned a coarse recyclable decision. This shows that even a strong pretrained model can produce high confidence for an unknown object when forced to choose among known labels. This supports the OpenWaste-HR research argument that local unknown evaluation, safe hierarchical policies, and manual-review routing are necessary for real-world waste classification.

## Backend and Frontend Best Policy Demo Test v1 Summary

This stage tested the live OpenWaste-HR prototype after integrating the current best pretrained safe hierarchical policy.

Created files:

* docs/methodology/backend_frontend_best_policy_demo_test_v1.md
* docs/supervisor_updates/backend_frontend_best_policy_demo_test_summary_v1.md
* tests/test_backend_frontend_best_policy_demo_docs.py

Test result:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   202 |
| Warnings          |     1 |
| Blocking failures |     0 |

Backend server result:

| Field           | Value                                             |
| --------------- | ------------------------------------------------- |
| Server          | uvicorn backend.app.main:app --reload --port 8000 |
| Backend URL     | http://127.0.0.1:8000                             |
| Startup result  | Application startup complete                      |
| Health endpoint | success                                           |
| Health status   | ok                                                |
| Service         | openwaste-hr-backend                              |
| Version         | backend_inference_endpoint_v1                     |

Backend prediction endpoint result:

| Field                  | Value                                         |
| ---------------------- | --------------------------------------------- |
| Request ID             | backend_best_policy_demo_001                  |
| Sample ID              | local_000001                                  |
| Image path             | ml/data/local_unknown/images/local_000001.jpg |
| Status                 | success                                       |
| Device                 | cuda                                          |
| Predicted fine label   | paper_cardboard                               |
| Max softmax confidence | 0.993654                                      |
| Top coarse label       | recyclable                                    |
| Top coarse confidence  | 0.999999                                      |
| Coarse margin          | 0.999999                                      |
| Decision type          | coarse_label                                  |
| Final label            | recyclable                                    |
| Final confidence       | 0.999999                                      |
| Decision reason        | coarse_fallback_stable                        |

Active policy thresholds:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

Frontend demo result:

| Field                      | Value                             |
| -------------------------- | --------------------------------- |
| Frontend server            | python -m http.server 5500        |
| Frontend URL               | http://127.0.0.1:5500             |
| Browser demo status        | displayed successful raw response |
| Frontend request ID        | frontend_demo_1782032776455       |
| Frontend decision type     | coarse_label                      |
| Frontend final label       | recyclable                        |
| Frontend final confidence  | 0.999999                          |
| Frontend policy thresholds | 0.995 / 0.8 / 0.15 / 0.9          |

Research note:

The live backend and frontend demo are now connected to the best current OpenWaste-HR policy. The demo image local_000001 is a rubber slipper / flip-flop, which is outside the known TrashNet-style training classes. The model still predicted a known fine label with high confidence and returned a coarse recyclable decision. This live demo therefore gives a useful thesis example showing why real-world waste classification needs local unknown evaluation, hierarchical fallback decisions, manual-review routing, and future active-learning correction.

## Human Correction Label Preparation v1 Summary

This stage prepared OpenWaste-HR for human correction labels and active learning v2.

Created files:

* docs/annotation/local_unknown_human_label_guide_v1.md
* docs/methodology/human_correction_label_preparation_v1.md
* docs/supervisor_updates/human_correction_label_preparation_summary_v1.md
* ml/data/splits/local_unknown_human_observation_notes_v1.csv
* ml/outputs/metrics/human_labelling_sheet_v1_working_review.csv
* tests/test_human_correction_label_preparation_docs.py

Purpose:

This stage documents how local unknown images should be reviewed by a human before being used for active learning v2.

Confirmed local unknown example:

| Sample ID    | Human Observation          | Taxonomy Status                | Recommended Action   |
| ------------ | -------------------------- | ------------------------------ | -------------------- |
| local_000001 | rubber slipper / flip-flop | outside_current_known_taxonomy | keep_as_unknown_test |

Research note:

The local_000001 image is a rubber slipper / flip-flop and is outside the current known fine-label taxonomy. However, the best pretrained safe policy predicted a known fine label with high confidence and returned a coarse recyclable decision. This supports the OpenWaste-HR argument that real-world waste classification requires local unknown evaluation, human correction labels, manual-review routing, and active learning support.

The working human review file is:

```text
ml/outputs/metrics/human_labelling_sheet_v1_working_review.csv
```

This file should be used for manual review instead of editing the original generated human labelling sheet.









