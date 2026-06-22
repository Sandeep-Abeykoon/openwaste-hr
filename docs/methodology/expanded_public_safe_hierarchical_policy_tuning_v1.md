# Expanded Public Safe Hierarchical Policy Tuning v1

## Purpose

This stage tunes a safe hierarchical decision policy for Baseline C.

Baseline C is the pretrained expanded public dataset model trained using combined TrashNet-style and RealWaste known samples.

## Baseline C Definition

```text id="ssg3xs"
Baseline C = pretrained expanded public dataset model
```

## Evaluation Goal

The goal is to tune a final OpenWaste-HR decision policy that balances known-class usefulness and unknown-sample safety.

The policy should avoid treating the model as a simple closed-set classifier. Instead, the system may output one of the following decisions:

| Decision        | Meaning                                            |
| --------------- | -------------------------------------------------- |
| fine label      | confident fine-grained waste class prediction      |
| coarse fallback | safer recyclable/organic/residual level prediction |
| manual review   | reject or unknown-like case requiring human review |

## Why This Stage Is Needed

Previous stages showed different strengths:

| Stage                                    | Main Finding                                                   |
| ---------------------------------------- | -------------------------------------------------------------- |
| expanded public closed-set evaluation    | strong six-class known classification performance              |
| expanded public reject-option evaluation | confidence threshold gave strongest selective known prediction |
| expanded public unknown evaluation       | energy score gave strongest unknown rejection                  |

Therefore, the final safe policy should not be based only on closed-set accuracy. It should balance:

* known coverage
* accepted prediction reliability
* local unknown rejection
* public unknown/future-class rejection
* hierarchical fallback behaviour

## Input Model

| Item          | Path                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| checkpoint    | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_best.pt            |
| class mapping | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_class_mapping.json |

## Input Data

| Input              | Manifest                                         |
| ------------------ | ------------------------------------------------ |
| known test data    | ml/data/splits/expanded_public_known_test_v1.csv |
| local unknown data | ml/data/splits/local_unknown_manifest_v1.csv     |

## Tuning Strategy

The tuning process searches candidate policy thresholds and evaluates each candidate on known and unknown data.

The policy considers:

| Parameter                   | Purpose                                                                            |
| --------------------------- | ---------------------------------------------------------------------------------- |
| fine confidence threshold   | controls when a fine label is safe enough                                          |
| coarse confidence threshold | controls when coarse fallback is allowed                                           |
| fine margin threshold       | checks whether the top fine prediction is sufficiently separated from alternatives |
| minimum coarse confidence   | prevents weak coarse predictions from being accepted                               |

## Metrics

The tuning process evaluates:

* known coverage
* known manual review rate
* accepted known prediction success
* local unknown manual review rate
* local unknown acceptance rate

## Research Meaning

This is the final policy-tuning stage for the expanded public model. It connects the technical implementation to the main research aim of OpenWaste-HR: a hierarchical, uncertainty-aware waste classifier that can avoid unsafe closed-set predictions.
