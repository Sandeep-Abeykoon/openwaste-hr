# OpenWaste-HR Reject-Option and Open-Set Evaluation Protocol v1

## Purpose

This protocol defines how OpenWaste-HR will evaluate safe prediction, unknown rejection, confidence calibration, and open-set behaviour.

The project is not only a closed-set waste classifier. It must show whether the model can safely classify known waste and reject unknown waste instead of forcing every image into cardboard, glass, metal, paper, or plastic.

## Known Classes

The known classes are:

- cardboard
- glass
- metal
- paper
- plastic

## Unknown Evaluation Classes

The main unknown evaluation classes are:

- textile / clothes / fabric
- biological / food organic

These classes must not be used as known training classes.

## Dataset Splits Required

The project must maintain separate splits:

- known_train
- known_val
- known_test
- unknown_val
- unknown_test

The unknown_val split is used only for threshold tuning.

The unknown_test split is used only for final unknown-rejection evaluation.

## Reject-Option Methods

The project will evaluate the following methods.

### Method 1: Softmax Confidence Threshold

The model predicts a fine class only if the maximum softmax confidence is above a selected threshold.

Example decision policy:

- if max softmax confidence >= threshold: accept fine-class prediction
- otherwise: reject as unknown / manual review

The threshold is not fixed manually. It must be selected using validation results.

### Method 2: Temperature-Scaled Confidence Threshold

Temperature scaling is applied to improve confidence calibration.

The same confidence-threshold decision is then applied after calibration.

This method tests whether better-calibrated confidence improves safe rejection and reduces overconfident wrong predictions.

### Method 3: Max-Logit Score

The model uses the highest raw logit value before softmax.

Known images are expected to produce stronger logits than unknown images.

Decision policy:

- if max logit >= threshold: accept prediction
- otherwise: reject as unknown / manual review

### Method 4: Energy Score

Energy score is used as an open-set score based on the model logits.

Energy-based scoring helps compare known and unknown separation without relying only on softmax confidence.

Decision policy:

- calculate energy score from logits
- compare against selected threshold
- accept known prediction or reject as unknown/manual review

### Method 5: Safe Final Policy

The final policy combines the strongest validation-performing method with a safe decision rule.

Possible decisions:

- accepted fine label
- uncertain manual review
- unknown / rejected

The final policy must be selected using validation data and then evaluated once on the final test data.

## Threshold Selection Rule

Thresholds must not be chosen randomly.

Thresholds will be selected using known_val and unknown_val only.

Candidate thresholds may be searched across score ranges, for example:

- confidence thresholds from 0.50 to 0.95
- max-logit thresholds from validation score distributions
- energy thresholds from validation score distributions

The selected threshold should balance:

- high known-class accuracy
- low known-class rejection
- high unknown rejection
- low confident wrong predictions

The selected threshold is then frozen and tested on known_test and unknown_test.

## Metrics

Closed-set metrics:

- accuracy
- macro-F1
- balanced accuracy
- confusion matrix

Reject-option and open-set metrics:

- unknown rejection rate
- known acceptance rate
- false rejection rate for known classes
- AUROC for known-vs-unknown separation
- ECE
- coverage-risk curve

## Active Learning Link

After each model stage:

1. Run the model on validation, unknown, and difficult/local samples.
2. Select uncertain, rejected, and confusing samples.
3. Create a human-review sheet.
4. Human-review selected samples.
5. Add only human-confirmed known samples to the next training set.
6. Keep true unknowns outside known training.
7. Retrain and compare before-vs-after performance.

## Important Research Rule

Unknown images must never be trained as a sixth class.

The system must learn to reject unknowns through scoring and thresholds, not by learning a direct unknown label.
