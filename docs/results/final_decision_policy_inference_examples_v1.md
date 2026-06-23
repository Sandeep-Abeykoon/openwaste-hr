# Final Decision Policy Inference Examples v1

## Purpose

This document records two example inference runs using the final Stage 4 model and the final energy-based reject-option decision policy.

The purpose is to confirm that the deployed inference script does not behave like a simple closed-set classifier. Instead, it applies the final decision policy:

```text
fine label if accepted as known
unknown/manual review if rejected
```

## Example 1: Known Test Image

| Item | Value |
|---|---|
| Image path | ml\data\raw\trashbox\plastic\plastic 1777.jpg |
| Model top-1 prediction | plastic |
| Raw confidence | 1.0000 |
| Temperature-scaled confidence | 0.9999 |
| Energy score | -16.1523 |
| Decision | known_fine_label |
| Accepted as known | true |
| Fine label shown | plastic |
| Coarse label shown | recyclable |

### User-facing message

```text
This item is likely plastic. It belongs to the recyclable category.
```

## Example 2: Unknown Test Image

| Item | Value |
|---|---|
| Image path | ml\data\raw\garbage_v2\clothes\clothes_319.jpg |
| Actual unknown type | textile/clothes |
| Internal model top-1 prediction | paper |
| Raw confidence | 0.6594 |
| Temperature-scaled confidence | 0.5049 |
| Energy score | -4.1324 |
| Decision | unknown_manual_review |
| Accepted as known | false |
| Fine label shown | none |
| Coarse label shown | none |

### User-facing message

```text
The system is not confident that this item belongs to the supported known classes. Please send it for manual review.
```

## Interpretation

The known plastic example was accepted correctly because its energy score was well below the final energy threshold.

The unknown clothes/textile example was rejected correctly because its energy score did not pass the known-class acceptance threshold. Although the internal top-1 model prediction was paper, this label was not shown as a confident user-facing result.

This confirms that the final inference pipeline applies the open-set reject option correctly.
