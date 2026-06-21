# OpenWaste-HR

## Project Title

**OpenWaste-HR: Hierarchical Open-Set Waste Classification with Reject/Manual-Review Option and Local Active Learning**

## Project Summary

OpenWaste-HR is a final-year research and prototype project for safer real-world waste classification.

The project combines:

* pretrained image classification
* hierarchical fine/coarse waste taxonomy
* reject/manual-review decision support
* local unknown evaluation
* safe threshold tuning
* backend/frontend prototype integration
* human-in-the-loop active learning

The main research argument is that waste classification should not be treated only as a closed-set image classification problem. In real-world use, local waste images may contain unfamiliar objects outside the training taxonomy. A normal classifier may still produce a high-confidence known label for these unknown objects.

OpenWaste-HR addresses this by allowing the system to return:

| Decision Type | Meaning                                   |
| ------------- | ----------------------------------------- |
| fine_label    | confident known fine-label prediction     |
| coarse_label  | safer broad fallback category             |
| manual_review | uncertain or unsafe case routed to review |

## Current Best System

The current best system is:

```text
Pretrained Safe Hierarchical Policy
```

| Item            | Value                                                                        |
| --------------- | ---------------------------------------------------------------------------- |
| Model           | pretrained transfer-learning model                                           |
| Checkpoint      | ml/outputs/checkpoints/pretrained_trashnet_v1/pretrained_trashnet_v1_best.pt |
| Decision policy | pretrained safe hierarchical policy                                          |

Selected thresholds:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.995000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.150000 |
| minimum_confidence_for_coarse | 0.900000 |

## Best Evaluation Result

| Metric                            |    Value |
| --------------------------------- | -------: |
| Known-test coverage               | 0.864583 |
| Accepted hierarchical reliability | 0.960843 |
| Local unknown manual-review rate  | 0.600000 |
| Local unknown acceptance rate     | 0.400000 |

## Closed-Set Model Improvement

| Model                    | Accuracy | Macro-F1 |
| ------------------------ | -------: | -------: |
| Scratch-trained baseline | 0.692708 | 0.645600 |
| Pretrained baseline      | 0.888000 | 0.851000 |

The pretrained model improves known-class classification substantially, but OpenWaste-HR also shows that high known-test accuracy alone is not enough for local unknown safety.

## Policy Comparison

| Policy                       | Known Coverage | Accepted Reliability | Local Unknown Manual Review | Local Unknown Acceptance |
| ---------------------------- | -------------: | -------------------: | --------------------------: | -----------------------: |
| Scratch safe hierarchical    |       0.658854 |             0.889328 |                    0.375000 |                 0.625000 |
| Pretrained hierarchical v1   |       0.976562 |             0.957333 |                    0.075000 |                 0.925000 |
| Pretrained safe hierarchical |       0.864583 |             0.960843 |                    0.600000 |                 0.400000 |

The pretrained safe hierarchical policy is the best current policy because it balances known-test coverage, accepted-decision reliability, and local unknown safety.

## Local Unknown Example

The live demo used:

```text
ml/data/local_unknown/images/local_000001.jpg
```

Human observation:

```text
rubber slipper / flip-flop
```

This object is outside the current known fine-label taxonomy.

The model/API result was:

| Field                  | Value           |
| ---------------------- | --------------- |
| Predicted fine label   | paper_cardboard |
| Max softmax confidence | 0.993654        |
| Decision type          | coarse_label    |
| Final label            | recyclable      |
| Final confidence       | 0.999999        |

This example shows why OpenWaste-HR needs local unknown evaluation and human-in-the-loop active learning.

## Active Learning v2

The reviewed local sample was assigned as:

| Field                   | Value                                   |
| ----------------------- | --------------------------------------- |
| Sample ID               | local_000001                            |
| Human observation       | rubber slipper / flip-flop              |
| Taxonomy status         | outside_current_known_taxonomy          |
| Recommended action      | keep_as_unknown_test                    |
| Active learning v2 role | unknown_test_and_future_class_candidate |

This prevents the unknown object from being incorrectly added to existing known training classes.

## Repository Structure

```text
backend/                 FastAPI backend prototype
frontend/                Simple frontend demo page
ml/
  configs/               YAML configuration files
  data/splits/           dataset manifests and reviewed local-label plans
  src/openwaste_hr/      training, evaluation, inference, and utility code
docs/
  methodology/           method notes and protocol files
  results/               evaluation reports and comparison summaries
  thesis/                thesis-ready chapter drafts and sections
  supervisor_updates/    supervisor summaries, demo script, and handover pack
tests/                   pytest validation tests
research_log.md          chronological project log
```

## Setup

Create and activate a Python virtual environment, then install the project dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Set the Python path before running project modules:

```powershell
$env:PYTHONPATH="ml/src;."
```

## Run Tests

Run the full test suite:

```powershell
$env:PYTHONPATH="ml/src;."
pytest
```

The latest full project test suite reached:

```text
248 passed, 1 warning
```

before the final README update step.

## Run Single-Image Inference

```powershell
$env:PYTHONPATH="ml/src;."
python -m openwaste_hr.inference.single_image_inference --config ml/configs/inference_pipeline.yaml --project-root . --image ml/data/local_unknown/images/local_000001.jpg --sample-id local_000001
```

## Run API Wrapper

```powershell
$env:PYTHONPATH="ml/src;."
python -m openwaste_hr.inference.api_wrapper --config ml/configs/prototype_api_wrapper.yaml --project-root . --image ml/data/local_unknown/images/local_000001.jpg --sample-id local_000001 --request-id best_policy_demo_001
```

## Run Backend

Open PowerShell in the project root:

```powershell
cd "D:\Github Repositories\openwaste-hr"
$env:PYTHONPATH="ml/src;."
uvicorn backend.app.main:app --reload --port 8000
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET
```

Prediction request:

```powershell
$body = @{
    request_id = "backend_best_policy_demo_001"
    sample_id = "local_000001"
    image_path = "ml/data/local_unknown/images/local_000001.jpg"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/api/inference/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

## Run Frontend Demo

Open another PowerShell window:

```powershell
cd "D:\Github Repositories\openwaste-hr\frontend"
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

Use this image path:

```text
ml/data/local_unknown/images/local_000001.jpg
```

## Key Thesis Files

| Purpose                       | File                                                            |
| ----------------------------- | --------------------------------------------------------------- |
| methodology chapter           | docs/thesis/methodology_chapter_consolidated_v1.md              |
| implementation chapter        | docs/thesis/implementation_chapter_draft_v1.md                  |
| evaluation chapter draft      | docs/thesis/evaluation_chapter_draft_v1.md                      |
| final evaluation update       | docs/thesis/evaluation_best_policy_active_learning_update_v1.md |
| active learning v2 section    | docs/thesis/active_learning_v2_section_v1.md                    |
| thesis assembly checklist     | docs/thesis/thesis_assembly_checklist_v1.md                     |
| final model/policy comparison | docs/results/final_model_policy_comparison_v1.md                |
| final evaluation summary      | docs/results/final_evaluation_summary_best_policy_v1.md         |
| supervisor handover pack      | docs/supervisor_updates/supervisor_handover_pack_v1.md          |
| supervisor demo script        | docs/supervisor_updates/supervisor_demo_script_v1.md            |

## Completed Pipeline

| Stage                                      | Status   |
| ------------------------------------------ | -------- |
| taxonomy and label map                     | complete |
| dataset manifest validation                | complete |
| TrashNet intake                            | complete |
| scratch baseline training                  | complete |
| reject-option baselines                    | complete |
| local unknown evaluation                   | complete |
| hierarchical policy                        | complete |
| safe policy tuning                         | complete |
| pretrained training                        | complete |
| pretrained reject/local unknown evaluation | complete |
| pretrained safe policy selection           | complete |
| backend/frontend prototype                 | complete |
| human correction seed                      | complete |
| active learning v2 dataset plan            | complete |
| thesis section updates                     | complete |
| supervisor handover pack                   | complete |

## Remaining Work

Recommended future work:

1. review the remaining active-learning candidate images
2. collect more local unknown images
3. add more public waste datasets
4. evaluate more pretrained architectures
5. fine-tune using reviewed active-learning data
6. improve frontend upload support
7. prepare final dissertation formatting and citations
8. deploy the backend/frontend prototype

## Terminology

Use these terms consistently:

| Use This Term                       | Avoid                     |
| ----------------------------------- | ------------------------- |
| local unknown dataset               | corrected unknown dataset |
| manual_review                       | reject only               |
| coarse fallback                     | wrong broad prediction    |
| pretrained safe hierarchical policy | final model only          |
| human-in-the-loop active learning   | manual checking only      |
| outside_current_known_taxonomy      | wrong label               |

## Current Status

OpenWaste-HR now has a complete v1 research and prototype pipeline.

The strongest current thesis message is:

```text
OpenWaste-HR improves waste classification by combining pretrained image recognition with hierarchical open-set decision-making, safe reject/manual-review behaviour, local unknown evaluation, and human-in-the-loop active learning.
```
