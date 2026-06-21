# Backend and Frontend Best Policy Demo Test Summary v1

## Purpose

This stage tests the live backend and frontend demo after integrating the best current OpenWaste-HR policy.

## Active System

```text
Pretrained Safe Hierarchical Policy
```

## Active Thresholds

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

## Components Tested

| Component           | Test                               |
| ------------------- | ---------------------------------- |
| backend             | health endpoint                    |
| backend             | prediction endpoint                |
| frontend            | browser demo                       |
| local unknown image | realistic unknown-object behaviour |

## Demo Image

```text
ml/data/local_unknown/images/local_000001.jpg
```

This image is a rubber slipper / flip-flop and is treated as a local unknown object.

## Expected Thesis Value

This test demonstrates that the selected research policy is connected to the real prototype.

It also provides a useful example of why high closed-set confidence is not enough for open-world waste classification.
