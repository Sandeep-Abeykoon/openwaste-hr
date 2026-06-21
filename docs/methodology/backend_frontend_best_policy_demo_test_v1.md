# Backend and Frontend Best Policy Demo Test v1

## Purpose

This stage tests the live OpenWaste-HR prototype after integrating the current best pretrained safe hierarchical policy.

The goal is to confirm that the backend and frontend demo are aligned with the best current research result.

## Active System

```text
Pretrained Safe Hierarchical Policy
```

## Active Checkpoint

```text
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

## Active Policy Thresholds

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

## Components Tested

| Component                   | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| backend health endpoint     | confirms backend server is running                     |
| backend prediction endpoint | confirms prediction API uses the active policy         |
| frontend demo page          | confirms user-facing prototype can submit a prediction |
| local unknown image test    | checks behaviour on a real local unknown object        |

## Test Image

The demo test uses:

```text
ml/data/local_unknown/images/local_000001.jpg
```

This image is a rubber slipper / flip-flop, which is not one of the known training classes. Therefore, it is useful for showing why local unknown testing is needed.

## Expected Behaviour

The model may still assign a known label with high confidence because it is trained using a limited closed-set label space. The important point is to document the final hierarchical decision and whether the prototype is using the correct best-policy thresholds.

## What This Stage Proves

This stage proves that the experimental result is connected to the working prototype.

The project is not only producing offline metrics. The selected pretrained safe hierarchical policy is also integrated into the backend/API/frontend workflow.

## Research Meaning

This test is useful for the thesis demonstration because it shows a realistic open-world limitation. A rubber slipper is outside the known TrashNet-style categories, but the model may still map it to a known class or broad category.

This supports the OpenWaste-HR argument that real-world waste systems need:

* local unknown evaluation
* hierarchical fallback decisions
* manual-review routing
* active learning for future correction
