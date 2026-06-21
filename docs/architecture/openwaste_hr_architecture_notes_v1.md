# OpenWaste-HR Architecture Diagram Notes v1

## Purpose

This document explains the architecture diagram source file:

```text
docs/architecture/openwaste_hr_architecture_v1.mmd
```

The diagram describes the current OpenWaste-HR prototype architecture from dataset preparation to frontend demonstration.

## Main Architecture Layers

| Layer                         | Purpose                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| Dataset and Data Management   | prepares known and local unknown image data                             |
| Model Training and Evaluation | trains and evaluates the baseline classifier                            |
| Hierarchical Decision Layer   | applies fine-label, coarse-label, or manual-review decision logic       |
| Local Active Learning Loop    | selects local images for human labelling and future dataset improvement |
| Inference Layer               | provides single-image, batch, and API-wrapper inference                 |
| Backend Layer                 | exposes the inference workflow through FastAPI endpoints                |
| Frontend Demo                 | allows browser-based demonstration of the final decision                |

## Dataset and Data Management

The data layer includes raw known datasets, the local unknown dataset, the dataset manifest builder, train/validation/test splits, and dataset inspection.

This layer is important because OpenWaste-HR depends on structured dataset records instead of only raw folders.

## Model Training and Evaluation

The model layer trains the MobileNetV3-style baseline classifier and evaluates it using closed-set metrics, reject-option baselines, and local unknown evaluation.

This shows the limitation of forced classification and motivates the safer decision layer.

## Hierarchical Decision Layer

The hierarchical decision layer is the main novelty of the prototype.

It allows the system to output:

| Decision Type | Meaning                                         |
| ------------- | ----------------------------------------------- |
| fine_label    | detailed known waste label                      |
| coarse_label  | safer broad fallback category                   |
| manual_review | uncertain or unsafe case routed to human review |

The selected safe policy uses stricter thresholds to reduce unsafe acceptance of unknown local images.

## Local Active Learning Loop

The active learning layer selects useful local unknown images for human review.

The workflow is:

```text
safe policy decisions → candidate selection → human labelling sheet → reviewed label processing → future dataset v2
```

This supports the project goal of improving the system using local feedback over time.

## Inference Layer

The inference layer turns the trained model and safe policy into usable prediction tools.

It includes:

| Component              | Purpose                                   |
| ---------------------- | ----------------------------------------- |
| single-image inference | process one image path                    |
| batch inference        | process a folder of images                |
| prototype API wrapper  | convert inference output into stable JSON |

## Backend Layer

The backend layer exposes the inference workflow through FastAPI.

Current endpoints:

| Method | Endpoint               | Purpose                                 |
| ------ | ---------------------- | --------------------------------------- |
| GET    | /health                | backend health check                    |
| POST   | /api/inference/predict | run OpenWaste-HR inference on one image |

## Frontend Demo

The frontend demo is a simple HTML, CSS, and JavaScript interface.

It sends an image path to the backend and displays:

* final label
* decision type
* decision reason
* confidence
* class probabilities
* raw backend JSON response

## Thesis Use

This architecture diagram can be used in the thesis implementation chapter to show how the system components connect.

Recommended caption:

**Figure X: OpenWaste-HR prototype architecture showing dataset preparation, hierarchical decision-making, active learning, inference, backend, and frontend demo layers.**

## Export Note

The `.mmd` file can be rendered using a Mermaid-compatible tool or editor extension and exported as PNG or SVG for the thesis.
