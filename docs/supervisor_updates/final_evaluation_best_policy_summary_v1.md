# Final Evaluation Best Policy Summary v1

## Purpose

This update summarises the current final evaluation position of OpenWaste-HR.

## Best Current System

```text
Pretrained Safe Hierarchical Policy
```

## Main Metrics

| Metric                            |    Value |
| --------------------------------- | -------: |
| Known-test coverage               | 0.864583 |
| Accepted hierarchical reliability | 0.960843 |
| Local unknown manual-review rate  | 0.600000 |
| Local unknown acceptance rate     | 0.400000 |

## Why This Is the Best Current Policy

The pretrained safe hierarchical policy improves the previous scratch safe policy and avoids the overly permissive behaviour of the first pretrained hierarchical policy.

| Policy                       | Known Coverage | Accepted Reliability | Local Unknown Manual Review |
| ---------------------------- | -------------: | -------------------: | --------------------------: |
| Scratch safe hierarchical    |       0.658854 |             0.889328 |                    0.375000 |
| Pretrained hierarchical v1   |       0.976562 |             0.957333 |                    0.075000 |
| Pretrained safe hierarchical |       0.864583 |             0.960843 |                    0.600000 |

## Active Learning v2 Link

The reviewed local unknown example was:

```text
local_000001 = rubber slipper / flip-flop
```

The active learning v2 plan keeps this sample as:

```text
unknown_test_and_future_class_candidate
```

It is not added to known training because it is outside the current known taxonomy.

## Thesis Message

The strongest current thesis message is:

```text
OpenWaste-HR improves waste classification by combining pretrained image recognition with hierarchical open-set decision-making, safe reject/manual-review behaviour, local unknown evaluation, and human-in-the-loop active learning.
```

## Next Stage

The next stage should update the implementation/evaluation chapter index or create a final thesis assembly checklist so the completed sections can be inserted into the main dissertation document.
