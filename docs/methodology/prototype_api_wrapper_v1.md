# Prototype API Wrapper v1

## Purpose

This stage creates a lightweight API-style wrapper for OpenWaste-HR inference.

The previous inference stages can process a single image or a folder of images. This wrapper converts the single-image inference result into a cleaner request/response format that can later be called by a backend service.

## Why This Is Needed

A production or prototype backend should not directly handle raw model internals. It should call a stable interface.

This wrapper provides that interface.

## Input Request Format

The wrapper accepts a request containing:

| Field      | Required | Meaning                            |
| ---------- | -------- | ---------------------------------- |
| image_path | yes      | project-relative path to the image |
| sample_id  | no       | optional custom sample ID          |

Example:

```json
{
  "image_path": "ml/data/local_unknown/images/local_000001.jpg",
  "sample_id": "local_000001"
}
```

## Output Response Format

The wrapper returns a structured response with:

| Section             | Meaning                                         |
| ------------------- | ----------------------------------------------- |
| status              | success or error                                |
| request             | original request information                    |
| prediction          | model prediction values                         |
| decision            | final OpenWaste-HR decision                     |
| class_probabilities | probability for each known fine label           |
| policy              | thresholds used by the safe hierarchical policy |

## Decision Values

| Decision Type | Meaning                                 |
| ------------- | --------------------------------------- |
| fine_label    | return a detailed known waste label     |
| coarse_label  | return a broader safe fallback category |
| manual_review | send the image to human/manual review   |

## Research Meaning

This stage moves OpenWaste-HR from offline scripts toward prototype integration.

The system now has a stable response format that can be used by a future backend, frontend, or demo application.
