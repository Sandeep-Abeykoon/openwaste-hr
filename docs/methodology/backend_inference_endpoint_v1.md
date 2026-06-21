# Backend Inference Endpoint Skeleton v1

## Purpose

This stage creates a lightweight backend endpoint skeleton for OpenWaste-HR inference.

The previous prototype API wrapper produced a clean request and response format in Python. This stage exposes that wrapper through a FastAPI backend route.

## Endpoints

| Method | Endpoint               | Purpose                                 |
| ------ | ---------------------- | --------------------------------------- |
| GET    | /health                | check whether the backend is running    |
| POST   | /api/inference/predict | run OpenWaste-HR inference on one image |

## Request Format

The prediction endpoint accepts:

```json
{
  "image_path": "ml/data/local_unknown/images/local_000001.jpg",
  "sample_id": "local_000001",
  "request_id": "demo_request_001"
}
```

## Response Format

The endpoint returns:

| Section             | Meaning                                                   |
| ------------------- | --------------------------------------------------------- |
| status              | success or error                                          |
| request_id          | request identifier                                        |
| request             | sample ID and image path                                  |
| prediction          | model prediction values                                   |
| decision            | final fine-label, coarse-label, or manual-review decision |
| class_probabilities | known fine-label probabilities                            |
| policy              | safe hierarchical thresholds                              |
| metadata            | device and pipeline version                               |

## Research Meaning

This stage moves OpenWaste-HR closer to a deployable prototype.

The model pipeline can now be called through a backend-style endpoint instead of only through command-line scripts. This supports future frontend integration and a complete project demonstration.
