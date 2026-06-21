# Best Pretrained Safe Policy Integration Summary v1

## Purpose

This stage updates the active prototype to use the current best OpenWaste-HR system.

## Active System

```text id="o8iu55"
Pretrained Safe Hierarchical Policy
```

## Active Checkpoint

```text id="hayu6x"
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

## Active Thresholds

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

## Updated Prototype Components

| Component              | Update                               |
| ---------------------- | ------------------------------------ |
| single-image inference | now uses pretrained safe policy      |
| batch inference        | now uses pretrained safe policy      |
| API wrapper            | now uses pretrained safe policy      |
| backend endpoint       | uses updated API wrapper config      |
| frontend demo          | displays current best policy wording |

## Why This Matters

The prototype now matches the best current research result.

The demo is no longer based on the earlier scratch-trained policy. It now uses the pretrained safe hierarchical policy that achieved:

| Metric                            |    Value |
| --------------------------------- | -------: |
| Known-test coverage               | 0.864583 |
| Accepted hierarchical reliability | 0.960843 |
| Local unknown manual-review rate  | 0.600000 |
| Local unknown acceptance rate     | 0.400000 |

## Next Stage

After this, the prototype should be tested again through:

1. single-image inference
2. prototype API wrapper
3. backend endpoint
4. frontend demo
