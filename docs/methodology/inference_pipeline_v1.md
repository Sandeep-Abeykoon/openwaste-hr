# Inference Pipeline v1

## Purpose

This stage creates the first single-image inference pipeline for OpenWaste-HR.

The pipeline loads a trained classifier checkpoint, processes one image, produces fine-label probabilities, applies the hierarchical decision policy, and returns the final system decision.

## Input

The input is a single image path.

Example:

```text
ml/data/local_unknown/images/local_000001.jpg
```

## Output

The inference pipeline produces:

| Output               | Meaning                                        |
| -------------------- | ---------------------------------------------- |
| predicted fine label | model’s top fine-label prediction              |
| fine confidence      | maximum softmax confidence                     |
| top coarse label     | aggregated coarse-level prediction             |
| coarse confidence    | summed probability of the top coarse group     |
| coarse margin        | separation between top and second coarse group |
| decision type        | fine_label, coarse_label, or manual_review     |
| final label          | final OpenWaste-HR output                      |
| decision reason      | explanation of why the decision was selected   |

## Decision Policy

The inference pipeline uses the selected safe hierarchical policy:

| Threshold                     | Value |
| ----------------------------- | ----: |
| fine_confidence_threshold     |  0.90 |
| coarse_confidence_threshold   |  0.80 |
| coarse_margin_threshold       |  0.65 |
| minimum_confidence_for_coarse |  0.65 |

## Decision Meaning

| Decision Type | Meaning                                                        |
| ------------- | -------------------------------------------------------------- |
| fine_label    | return a detailed waste label                                  |
| coarse_label  | return a broader category because fine prediction is uncertain |
| manual_review | route the image to human/manual review                         |

## Research Meaning

This stage turns the experimental model into a usable OpenWaste-HR inference component.

It demonstrates that the project is not only producing offline metrics. It can also process a new image and return a safe hierarchical decision suitable for later backend or frontend integration.
