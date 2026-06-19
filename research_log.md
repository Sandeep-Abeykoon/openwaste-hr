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
| 2026-06-19 | Ran local unknown evaluation | Evaluated softmax confidence, maximum logit, and energy-score rejection on 42 local phone-captured unknown images |

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

### Actual Local Unknown Dataset v1

| Item | Value |
|---|---:|
| Local captured images | 42 |
| Usage | unknown_test |
| Fine label | unknown |
| Coarse label | unknown |
| Source dataset | local_phone_images |

Research note:

The first local unknown dataset contains 42 phone-captured difficult or locally shifted waste/object images. These images are not used for closed-set training. They are reserved for unknown/manual-review evaluation, where the system should ideally reject uncertain or out-of-distribution inputs instead of forcing a known fine label.

## Local Unknown Evaluation v1 Summary

This stage evaluates the first local unknown dataset.

The local unknown dataset contains 42 phone-captured images.

For this evaluation, rejection/manual review is treated as desirable behaviour. Acceptance as a known fine label is treated as a false acceptance.

Evaluated methods:

| Method | Score | Accept Rule |
|---|---|---|
| Confidence | max softmax confidence | accept if score >= selected threshold |
| Maximum Logit Score | max(logits) | accept if score >= selected threshold |
| Energy Score | -T * logsumexp(logits / T) | accept if score <= selected threshold |

Generated files:

- docs/results/local_unknown_evaluation_v1_report.md
- docs/results/figures/local_unknown_rejection_rates_v1.png
- docs/results/figures/local_unknown_accepted_label_distribution_v1.png
- ml/outputs/metrics/local_unknown_predictions_v1.csv
- ml/outputs/metrics/local_unknown_method_decisions_v1.csv
- ml/outputs/metrics/local_unknown_evaluation_metrics_v1.json
- ml/outputs/metrics/local_unknown_accepted_label_distribution_v1.csv

Research note:

This is the first true unknown/manual-review evaluation in the project. It tests the main OpenWaste-HR motivation more directly than closed-set accuracy because the system is evaluated on local unknown images not used in training.