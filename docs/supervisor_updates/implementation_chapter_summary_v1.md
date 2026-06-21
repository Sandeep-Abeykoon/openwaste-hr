# Implementation Chapter Summary v1

## What Was Implemented

OpenWaste-HR has been implemented as a working hierarchical open-set waste classification prototype.

The implemented workflow is:

```text
dataset preparation → baseline model → reject option → hierarchical decision policy → safe tuning → active learning → inference → backend → frontend
```

## Main Implementation Sections

| Section             | Summary                                                         |
| ------------------- | --------------------------------------------------------------- |
| Taxonomy            | fine labels, coarse labels, unknown, and manual-review concepts |
| Dataset pipeline    | manifest builder, split files, image inspection                 |
| Baseline model      | MobileNetV3-style closed-set classifier                         |
| Reject baselines    | confidence, max-logit, and energy scoring                       |
| Hierarchical policy | fine-label, coarse-label, or manual-review output               |
| Safe tuning         | stricter thresholds for safer unknown handling                  |
| Active learning     | selects local unknown candidates for human review               |
| Human labelling     | creates annotation sheet and processing workflow                |
| Inference           | single-image and batch inference                                |
| Backend             | FastAPI endpoint for prediction                                 |
| Frontend            | simple browser demo page                                        |
| Testing             | automated tests across all main components                      |

## Key Results to Mention

| Result                                             |      Value |
| -------------------------------------------------- | ---------: |
| Closed-set test accuracy                           |   0.692708 |
| Confidence reject local unknown rejection rate     |   0.350000 |
| Safe hierarchical local unknown manual-review rate |   0.375000 |
| Active learning candidates selected                |         20 |
| Human labelling sheet rows                         |         20 |
| Latest full test suite                             | 121 passed |
| Demo prediction                                    |    plastic |
| Demo decision type                                 | fine_label |
| Demo confidence                                    |   0.962933 |

## Core Novelty

The project should be presented as:

```text
Hierarchical open-set waste classification with reject option and local active learning
```

The main strength is not just the CNN classifier. The stronger contribution is the decision workflow that avoids unsafe forced predictions by using:

* fine-label prediction
* coarse fallback
* manual-review routing
* local unknown evaluation
* active learning candidate selection

## Current Prototype Demo

The prototype currently supports:

```text
frontend → FastAPI backend → API wrapper → inference pipeline → hierarchical decision → frontend display
```

Demo input:

```text
ml/data/local_unknown/images/local_000001.jpg
```

Demo output:

| Field         | Value      |
| ------------- | ---------- |
| pred_label    | plastic    |
| confidence    | 0.962933   |
| decision_type | fine_label |
| final_label   | plastic    |

## Current Limitations

| Limitation                       | Explanation                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| trained labels are limited       | current classifier uses five known fine labels                      |
| local unknown acceptance remains | safe policy improves but does not solve unknown handling completely |
| human review is pending          | active learning sheet is ready but not filled                       |
| frontend has no upload yet       | current frontend uses project-relative image paths                  |
| stronger model is future work    | pretrained training may improve performance                         |

## Suggested Next Step

Use this draft as the base for the thesis implementation chapter, then add diagrams and screenshots from the prototype.
