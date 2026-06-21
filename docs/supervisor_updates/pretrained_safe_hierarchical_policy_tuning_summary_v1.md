# Safe Pretrained Hierarchical Policy Tuning Summary v1

## Purpose

This stage tunes a safer hierarchical policy for the pretrained transfer-learning model.

## Why This Stage Is Needed

The pretrained hierarchical policy produced very strong known-test results, but it still accepted many local unknown images.

Previous pretrained hierarchical result:

| Metric                           |    Value |
| -------------------------------- | -------: |
| Known-test coverage              | 0.976562 |
| Known accepted reliability       | 0.957333 |
| Local unknown manual-review rate | 0.075000 |
| Local unknown acceptance rate    | 0.925000 |

This means tuning is needed to improve local unknown safety.

## Config

```text id="2pekys"
ml/configs/pretrained_safe_hierarchical_policy_tuning.yaml
```

## Main Idea

The tuning process searches stricter thresholds for:

* fine_label acceptance
* coarse_label fallback
* coarse margin
* minimum confidence for coarse fallback

## What This Stage Should Produce

The stage should produce:

| Output                          | Purpose                                |
| ------------------------------- | -------------------------------------- |
| threshold sweep                 | compare candidate policies             |
| selected safe pretrained policy | best current safety-coverage trade-off |
| known-test decisions            | evaluate known decision quality        |
| local unknown decisions         | evaluate local unknown handling        |
| decision distribution           | show fine/coarse/manual-review balance |
| thesis report                   | document the result                    |

## Next Stage

After this stage, we should create a final model/policy comparison table comparing:

1. scratch baseline
2. scratch safe hierarchical policy
3. pretrained baseline
4. pretrained reject-option methods
5. pretrained safe hierarchical policy
