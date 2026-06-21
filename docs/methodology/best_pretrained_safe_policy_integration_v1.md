# Best Pretrained Safe Policy Integration v1

## Purpose

This stage updates the active OpenWaste-HR prototype to use the current best model and decision policy.

The current best system is:

```text id="b2ozcn"
Pretrained Safe Hierarchical Policy
```

## Integrated Model

The active inference pipeline now uses:

```text id="cz1em4"
ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt
```

This checkpoint belongs to Baseline B, the pretrained transfer-learning model.

## Integrated Policy

The active prototype now uses the selected safe pretrained hierarchical policy:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

## Files Updated

| File                                     | Purpose                                                         |
| ---------------------------------------- | --------------------------------------------------------------- |
| ml/configs/inference_pipeline.yaml       | single-image inference now uses the best pretrained safe policy |
| ml/configs/batch_inference_pipeline.yaml | batch inference now uses the best pretrained safe policy        |
| ml/configs/prototype_api_wrapper.yaml    | backend/API wrapper now uses the best pretrained safe policy    |
| frontend/index.html                      | frontend text now identifies the current best policy            |

## Why This Integration Is Needed

Earlier stages proved that the pretrained safe hierarchical policy is the best current OpenWaste-HR policy.

Its result was:

| Metric                            |    Value |
| --------------------------------- | -------: |
| Known-test coverage               | 0.864583 |
| Accepted hierarchical reliability | 0.960843 |
| Local unknown manual-review rate  | 0.600000 |
| Local unknown acceptance rate     | 0.400000 |

This stage makes the working prototype reflect that result.

## Research Meaning

This integration connects the experimental result to the usable prototype.

The frontend, backend, API wrapper, and inference configs are now aligned with the best current research result. This means the demo no longer uses the earlier scratch-trained policy; it uses the strongest current OpenWaste-HR decision workflow.
