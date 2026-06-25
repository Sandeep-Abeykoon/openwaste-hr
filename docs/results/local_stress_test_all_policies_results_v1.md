# Extended Local and Out-of-Taxonomy Stress-Test Evaluation: All Policies v1

## Purpose

This report evaluates the final OpenWaste-HR decision pipeline on an extended stress-test set containing local household waste images and additional vegetation images.

The purpose of this test is not to train the model, tune thresholds, or replace the main public benchmark. Instead, it checks whether the final manual-review policy behaves safely on real or deployment-oriented images that are outside or uncertain relative to the supported known taxonomy.

The stress-test images were not used for:

```text
training
validation
threshold tuning
fusion-gate training
model selection
```

They are used only as an independent stress-test after the final policy was selected.

---

## Stress-Test Dataset

| Property | Value |
|---|---:|
| Stress-test images | 462 |
| Source | Self-collected local images and additional vegetation images |
| Intended role | Independent out-of-taxonomy stress test |
| Expected status | Outside or uncertain relative to known training classes |

The known classifier taxonomy contains only:

```text
cardboard, glass, metal, paper, plastic
```

Therefore, vegetation and other unsupported local items should ideally be routed to manual review instead of being forced into a known recyclable label.

---

## Policies Compared

The same 462 stress-test images were evaluated using the following decision policies:

| Policy | Description |
|---|---|
| Raw classifier | Always returns one of the five known classes |
| Confidence threshold | Rejects low softmax-confidence predictions |
| Temperature-scaled confidence | Rejects low calibrated-confidence predictions |
| Max-logit | Rejects based on maximum logit score |
| Energy | Rejects using energy-based scoring |
| Fusion Gate v1 score-only | Uses score-level uncertainty features |
| Mahalanobis-only | Uses feature-distance knownness alone |
| Fusion Gate v2 + Mahalanobis | Final selected policy using score and feature-space evidence |

---

## Stress-Test Results

| Policy | Accepted as Known | Manual Review | Accepted-as-Known Rate | Manual-Review Rate |
|---|---:|---:|---:|---:|
| Raw classifier | 462/462 | 0/462 | 1.0000 | 0.0000 |
| Confidence threshold | 25/462 | 437/462 | 0.0541 | 0.9459 |
| Temperature-scaled confidence | 34/462 | 428/462 | 0.0736 | 0.9264 |
| Max-logit | 25/462 | 437/462 | 0.0541 | 0.9459 |
| Energy | 25/462 | 437/462 | 0.0541 | 0.9459 |
| Fusion Gate v1 score-only | 21/462 | 441/462 | 0.0455 | 0.9545 |
| Mahalanobis-only | 425/462 | 37/462 | 0.9199 | 0.0801 |
| Fusion Gate v2 + Mahalanobis | 13/462 | 449/462 | 0.0281 | 0.9719 |

---

## Forced Closed-Set Predictions

The raw classifier forced every stress-test image into one of the five known classes.

| Internal Top-1 Class | Count |
|---|---:|
| cardboard | 70 |
| plastic | 82 |
| paper | 20 |
| metal | 108 |
| glass | 182 |

This demonstrates the closed-set limitation clearly: without a reject option, every unsupported image still receives a known recyclable label.

---

## Final Fusion Gate v2 Stress-Test Behaviour

The final Fusion Gate v2 + Mahalanobis policy routed most stress-test images to manual review.

| Final Fusion Gate v2 Result | Count | Rate |
|---|---:|---:|
| Manual review | 449/462 | 0.9719 |
| Accepted as known | 13/462 | 0.0281 |

The 13 accepted cases were manually inspected from the exported accepted-case sheet. They belonged to the vegetation stress-test subset and were mainly accepted as glass or metal. These are treated as remaining hard false acceptances in the stress-test analysis.

---

## Interpretation

The raw classifier accepted all 462 stress-test images as known classes. This means the closed-set model alone would force every unsupported input into one of the five supported labels.

The final Fusion Gate v2 + Mahalanobis policy accepted only 13 out of 462 stress-test images and routed 449 images to manual review.

This is a safer outcome for deployment because the stress-test images were intended to represent outside or uncertain inputs. For such inputs, manual review is preferable to forcing a potentially incorrect known label.

---

## Mahalanobis-Only Finding

The Mahalanobis-only policy accepted 425 out of 462 stress-test images as known. This confirms the earlier quantitative result that Mahalanobis feature-distance alone is not reliable enough as the final reject-option policy.

However, when Mahalanobis evidence is combined with score-level signals inside Fusion Gate v2, the final policy becomes much safer on the stress-test set.

This supports the final design decision:

```text
Mahalanobis alone is weak,
but Mahalanobis is useful as complementary evidence inside Fusion Gate v2.
```

---

## Research Meaning

The extended stress test strengthens the final research story.

It shows that:

1. the raw closed-set classifier forces all stress-test images into known labels,
2. single-score rejectors reduce unsafe forced predictions,
3. Mahalanobis-only is not safe enough by itself,
4. the final Fusion Gate v2 policy routes 449 out of 462 stress-test images to manual review,
5. the remaining 13 accepted cases are hard false acceptances that should be discussed as a limitation.

The main public benchmark remains the primary evaluation. The extended stress test is used as additional deployment-oriented evidence.

---

## Thesis-Ready Summary

A thesis-ready summary is:

```text
An extended out-of-taxonomy stress-test set of 462 images was used as additional deployment-oriented evidence. The set included self-collected local household images and additional vegetation images, none of which were used for training, validation, threshold tuning, or fusion-gate training. The raw closed-set classifier forced all 462 images into one of the five known recyclable classes. In contrast, the final Fusion Gate v2 + Mahalanobis policy routed 449 images to manual review and accepted only 13 as known, giving a manual-review rate of 0.9719 and an accepted-as-known rate of 0.0281. Manual inspection showed that the remaining accepted cases were vegetation images mainly predicted as glass or metal, indicating a small number of hard false acceptances under visually confusing out-of-taxonomy conditions.
```

---

## Output Files

| Purpose | File |
|---|---|
| All-policy predictions | ml/outputs/local_stress_test/all_policies_v1/local_stress_test_all_policy_predictions_v1.csv |
| All-policy summary CSV | ml/outputs/local_stress_test/all_policies_v1/local_stress_test_all_policy_summary_v1.csv |
| All-policy summary JSON | ml/outputs/local_stress_test/all_policies_v1/local_stress_test_all_policy_summary_v1.json |
| Fusion Gate v2 accepted cases | ml/outputs/local_stress_test/all_policies_v1/fusion_gate_v2_accepted_cases_for_manual_review_v1.csv |

---

## Status

Extended local and out-of-taxonomy stress-test evaluation is complete.
