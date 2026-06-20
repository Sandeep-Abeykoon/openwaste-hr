# Batch Inference Pipeline v1

## Purpose

This stage creates the first batch inference pipeline for OpenWaste-HR.

The single-image inference pipeline processes one image at a time. The batch inference pipeline processes a folder of images and produces a CSV containing model predictions and final hierarchical decisions for each image.

## Input

The default input folder is:

```text
ml/data/local_unknown/images
```

The pipeline searches for image files using these extensions:

* .jpg
* .jpeg
* .png

## Output

The batch inference pipeline produces:

* batch_inference_results_v1.csv
* batch_inference_summary_v1.json
* batch_inference_report_v1.md

## Output Columns

| Column                       | Meaning                                        |
| ---------------------------- | ---------------------------------------------- |
| sample_id                    | image filename without extension               |
| image_path                   | project-relative image path                    |
| pred_label                   | model’s top fine-label prediction              |
| max_softmax_confidence       | confidence of the top fine prediction          |
| top_coarse_label             | aggregated coarse-label prediction             |
| top_coarse_confidence        | confidence of the top coarse group             |
| coarse_margin                | separation between top and second coarse group |
| hierarchical_decision_type   | fine_label, coarse_label, or manual_review     |
| hierarchical_final_label     | final OpenWaste-HR output                      |
| hierarchical_decision_reason | reason for the final decision                  |

## Decision Policy

The pipeline uses the selected safe hierarchical decision policy:

| Threshold                     | Value |
| ----------------------------- | ----: |
| fine_confidence_threshold     |  0.90 |
| coarse_confidence_threshold   |  0.80 |
| coarse_margin_threshold       |  0.65 |
| minimum_confidence_for_coarse |  0.65 |

## Research Meaning

This stage moves OpenWaste-HR closer to a usable prototype.

Instead of testing one image manually, the system can now process a batch of local images and export structured decisions for analysis, reporting, manual review, or future backend integration.
