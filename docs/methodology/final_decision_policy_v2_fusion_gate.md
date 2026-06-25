# Final Decision Policy v2: Fusion Gate with Mahalanobis

## Purpose

This document defines the final decision policy used by OpenWaste-HR after the reject-option extension.

The earlier final policy used a single energy threshold. That policy was effective, but the Mahalanobis-enhanced Fusion Gate v2 produced a stronger open-set result by combining score-level and feature-space evidence.

Therefore, the final research policy is updated to use Fusion Gate v2.

## Final System Overview

The final OpenWaste-HR decision pipeline is:

```text
input image
→ Stage 4 MobileNetV3 classifier
→ logits and class probabilities
→ temperature-scaled confidence for display
→ Mahalanobis feature-distance score
→ Fusion Gate v2 knownness score
→ accept known class or send to manual review
```

## Known Classes

The model is trained only on the five known recyclable classes:

| Known class |
|---|
| cardboard |
| glass |
| metal |
| paper |
| plastic |

These classes map to the coarse category:

```text
recyclable
```

## Unknown Classes

The unknown classes used for validation and testing are:

| Unknown class |
|---|
| biological |
| textile |

These unknown classes are not trained as a sixth class. They are used only for threshold selection and final open-set evaluation.

## Fusion Gate v2 Features

Fusion Gate v2 uses seven features:

| Feature | Description |
|---|---|
| confidence | Top-1 softmax confidence |
| temperature_scaled_confidence | Calibrated confidence used for safer display |
| max_logit | Maximum raw logit |
| energy | Energy-based open-set score |
| softmax_margin | Difference between top-1 and top-2 probabilities |
| softmax_entropy | Softmax uncertainty |
| mahalanobis_knownness | Feature-space knownness based on Mahalanobis distance |

## Final Decision Rule

The final Fusion Gate v2 threshold is:

```text
0.6314586412215439
```

The rule is:

```text
if fusion_knownness_score >= 0.6314586412215439:
    accept predicted known class
else:
    send image to manual review as unknown
```

## User-Facing Behaviour

If the item is accepted as known:

```text
This item is likely [predicted class]. It belongs to the recyclable category.
```

If the item is rejected:

```text
The system is not confident that this item belongs to the supported recyclable classes.
Please send it for manual review.
```

The rejected item still stores the internal top-1 prediction for audit purposes, but the system does not show this prediction to the user as a final label.

## Final Test Results

| Metric | Value |
|---|---:|
| Known test rows | 3,426 |
| Unknown test rows | 1,660 |
| Known coverage | 0.7656 |
| Unknown rejection rate | 0.9337 |
| False acceptance rate | 0.0663 |
| Accepted-known accuracy | 0.9752 |
| AUROC known vs unknown | 0.9269 |

## Comparison with Energy-Only Policy

| Method | AUROC | Known coverage | Unknown rejection | FAR | Accepted-known accuracy |
|---|---:|---:|---:|---:|---:|
| Energy-only policy | 0.8789 | 0.7665 | 0.8500 | 0.1500 | 0.9791 |
| Fusion Gate v2 + Mahalanobis | 0.9269 | 0.7656 | 0.9337 | 0.0663 | 0.9752 |

## Interpretation

Fusion Gate v2 substantially improves unknown rejection while preserving almost the same known coverage as the energy-only baseline.

Compared with the energy-only policy:

- AUROC increased from 0.8789 to 0.9269.
- Unknown rejection increased from 0.8500 to 0.9337.
- False acceptance rate decreased from 0.1500 to 0.0663.
- Known coverage stayed almost unchanged.
- Accepted-known accuracy remained high.

This supports the final research claim that safer open-world waste classification benefits from combining multiple uncertainty signals rather than relying only on softmax confidence or a single energy threshold.

## Final Selected Policy

| Component | Selected method |
|---|---|
| Image classifier | Stage 4 MobileNetV3 |
| Confidence display | Temperature-scaled confidence |
| Reject-option layer | Fusion Gate v2 |
| Feature-space evidence | Mahalanobis knownness |
| Final threshold | 0.6315 |
| Rejected item action | Manual review |

## Status

This is the final recommended decision policy for OpenWaste-HR.
