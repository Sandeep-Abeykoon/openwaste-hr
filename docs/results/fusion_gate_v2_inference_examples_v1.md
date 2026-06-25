# Final Fusion Gate v2 Inference Examples v1

## Purpose

This report records two single-image inference examples using the final OpenWaste-HR decision policy.

The final policy uses:

```text
Stage 4 MobileNetV3 classifier
+ temperature-scaled confidence
+ Mahalanobis feature-distance
+ Fusion Gate v2 knownness score
```

The aim is to show how the deployed system behaves for both a known recyclable item and an unknown/manual-review item.

## Final Decision Rule

The Fusion Gate v2 threshold is:

```text
0.6314586412215439
```

The rule is:

```text
if fusion_knownness_score >= 0.6314586412215439:
    accept predicted known class
else:
    send image to manual review
```

## Example 1: Known Plastic Image

| Field | Value |
|---|---|
| Image path | ml/data/raw/trashbox/plastic/plastic 1777.jpg |
| Internal top-1 prediction | plastic |
| Raw confidence | 1.0000 |
| Temperature-scaled confidence | 0.9999 |
| Energy | -16.1523 |
| Mahalanobis nearest class | plastic |
| Fusion knownness score | 0.9969 |
| Fusion threshold | 0.6315 |
| Accepted as known | true |
| Final decision type | known_fine_label |
| User-visible label | plastic |
| Coarse label | recyclable |

User-facing message:

```text
This item is likely plastic. It belongs to the recyclable category.
```

## Example 2: Unknown Textile Image

| Field | Value |
|---|---|
| Image path | ml/data/raw/garbage_v2/clothes/clothes_319.jpg |
| Actual unknown type | textile / clothes |
| Internal top-1 prediction | paper |
| Raw confidence | 0.6594 |
| Temperature-scaled confidence | 0.5049 |
| Energy | -4.1324 |
| Mahalanobis nearest class | plastic |
| Fusion knownness score | 0.0713 |
| Fusion threshold | 0.6315 |
| Accepted as known | false |
| Final decision type | unknown_manual_review |
| User-visible label | manual_review_required |
| Internal prediction shown to user | false |

User-facing message:

```text
The system is not confident that this item belongs to the supported recyclable classes. Please send it for manual review.
```

## Interpretation

The known plastic example was accepted correctly as a recyclable plastic item.

The unknown clothes image produced an internal top-1 prediction of paper, but the Fusion Gate v2 knownness score was far below the threshold. Therefore, the system rejected the image as unknown and sent it to manual review instead of incorrectly showing paper to the user.

This demonstrates the main safety purpose of the final decision policy: the model can still log its internal top-1 prediction, but the user-facing output is blocked when the fusion gate decides that the input does not safely belong to the supported known classes.

## Status

The final Fusion Gate v2 inference behaviour is working as expected.
