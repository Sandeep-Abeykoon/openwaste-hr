# Final README Update Summary v1

## Purpose

This stage updates the main project README so the repository is understandable as a final-year research project and working prototype.

## README Focus

The README now explains:

* project title
* research problem
* current best system
* best policy metrics
* local unknown example
* active learning v2 result
* repository structure
* setup commands
* inference commands
* backend/frontend demo commands
* thesis files
* completed pipeline
* remaining work

## Current Best System

```text
Pretrained Safe Hierarchical Policy
```

| Metric                           |    Value |
| -------------------------------- | -------: |
| Known-test coverage              | 0.864583 |
| Accepted reliability             | 0.960843 |
| Local unknown manual-review rate | 0.600000 |
| Local unknown acceptance rate    | 0.400000 |

## Key Example

```text
local_000001 = rubber slipper / flip-flop
model prediction = paper_cardboard
decision = coarse_label recyclable
human status = outside_current_known_taxonomy
active learning v2 role = unknown_test_and_future_class_candidate
```

## Why This Matters

The README now presents OpenWaste-HR as a complete research and prototype pipeline, not just a code repository.

It makes the final project easier for a supervisor, marker, or future reviewer to understand.
