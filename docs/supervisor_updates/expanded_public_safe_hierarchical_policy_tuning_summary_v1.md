# Expanded Public Safe Hierarchical Policy Tuning Summary v1

## Purpose

This stage tunes the safe hierarchical decision policy for the expanded public pretrained model.

## Baseline C

```text id="uig6sx"
Baseline C = pretrained expanded public dataset model
```

## Why This Matters

The expanded public model has already been evaluated for:

| Evaluation                               | Result Summary                      |
| ---------------------------------------- | ----------------------------------- |
| closed-set known classification          | strong six-class known performance  |
| reject-option known selective prediction | confidence threshold performed best |
| local and public unknown rejection       | energy score performed best         |

The safe hierarchical policy tuning stage combines these ideas into a final decision policy.

## Policy Outputs

| Output          | Meaning                               |
| --------------- | ------------------------------------- |
| fine label      | high-confidence fine class prediction |
| coarse fallback | safer higher-level waste category     |
| manual review   | uncertain or unknown-like sample      |

## Input Data

| Input              | Manifest                                         |
| ------------------ | ------------------------------------------------ |
| known test data    | ml/data/splits/expanded_public_known_test_v1.csv |
| local unknown data | ml/data/splits/local_unknown_manifest_v1.csv     |

## Research Point

This stage determines whether the expanded public pretrained model can become the final best OpenWaste-HR policy after combining hierarchical fallback and manual-review rejection.
