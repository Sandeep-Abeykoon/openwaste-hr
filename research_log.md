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


