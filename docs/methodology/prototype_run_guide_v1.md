# OpenWaste-HR Prototype Run Guide v1

## Purpose

This guide explains how to run the current OpenWaste-HR prototype.

The current prototype supports:

* trained baseline image classifier
* confidence-based rejection
* hierarchical fine/coarse/manual-review decision policy
* local unknown dataset evaluation
* active learning candidate selection
* human labelling sheet workflow
* single-image inference
* batch inference
* prototype API wrapper
* FastAPI backend endpoint
* simple frontend demo page

## 1. Activate the Environment

From the project root:

```powershell
cd "D:\Github Repositories\openwaste-hr"
```

Activate the virtual environment if needed:

```powershell
.venv\Scripts\activate
```

Set the Python path:

```powershell
$env:PYTHONPATH="ml/src;."
```

## 2. Run the Test Suite

Run:

```powershell
pytest tests/test_taxonomy.py tests/test_manifest.py tests/test_trashnet_manifest_builder.py tests/test_image_dataset.py tests/test_dataset_inspection.py tests/test_label_encoding.py tests/test_torch_dataset.py tests/test_baseline_model.py tests/test_classification_metrics.py tests/test_selective_metrics.py tests/test_open_set_scores.py tests/test_local_unknown_manifest_builder.py tests/test_unknown_metrics.py tests/test_hierarchical_decision.py tests/test_hierarchical_policy_tuning.py tests/test_active_learning_selection.py tests/test_human_labelling_sheet.py tests/test_reviewed_label_processing.py tests/test_inference_pipeline.py tests/test_batch_inference_pipeline.py tests/test_api_wrapper.py tests/test_backend_inference_endpoint.py tests/test_frontend_demo_files.py
```

Expected result:

```text
110 passed
```

A single FastAPI/TestClient deprecation warning is acceptable.

## 3. Check the Model Checkpoint

The inference pipeline expects the trained checkpoint at:

```text
ml/outputs/checkpoints/baseline_trashnet_v1/baseline_trashnet_best.pt
```

Check it with:

```powershell
dir ml\outputs\checkpoints\baseline_trashnet_v1
```

Expected files include:

```text
baseline_trashnet_best.pt
baseline_trashnet_final.pt
baseline_trashnet_class_mapping.json
```

## 4. Run Single-Image Inference

Run:

```powershell
python -m openwaste_hr.inference.single_image_inference --config ml/configs/inference_pipeline.yaml --project-root . --image ml/data/local_unknown/images/local_000001.jpg --sample-id local_000001
```

Expected decision for the demo image:

```text
pred_label: plastic
hierarchical_decision_type: fine_label
hierarchical_final_label: plastic
```

Generated files:

```text
ml/outputs/metrics/single_image_inference_result_v1.json
ml/outputs/metrics/single_image_inference_result_v1.md
```

## 5. Run Batch Inference

Run:

```powershell
python -m openwaste_hr.inference.batch_inference --config ml/configs/batch_inference_pipeline.yaml --project-root .
```

Expected output format:

```text
Batch inference completed successfully.
Images processed: ...
```

Generated files:

```text
ml/outputs/metrics/batch_inference_results_v1.csv
ml/outputs/metrics/batch_inference_summary_v1.json
ml/outputs/metrics/batch_inference_report_v1.md
```

## 6. Run the Prototype API Wrapper

Run:

```powershell
python -m openwaste_hr.inference.api_wrapper --config ml/configs/prototype_api_wrapper.yaml --project-root . --image ml/data/local_unknown/images/local_000001.jpg --sample-id local_000001 --request-id demo_request_001
```

Expected response:

```text
status: success
decision_type: fine_label
final_label: plastic
```

Generated files:

```text
ml/outputs/metrics/prototype_api_response_v1.json
ml/outputs/metrics/prototype_api_response_v1.md
```

## 7. Start the Backend

Run:

```powershell
$env:PYTHONPATH="ml/src;."
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Expected backend startup:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

Keep this terminal open.

## 8. Test the Backend Health Endpoint

Open a second PowerShell terminal and run:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET
```

Expected output:

```text
status: ok
service: openwaste-hr-backend
version: backend_inference_endpoint_v1
```

## 9. Test the Backend Prediction Endpoint

In the second terminal, run:

```powershell
$body = @{
    image_path = "ml/data/local_unknown/images/local_000001.jpg"
    sample_id = "local_000001"
    request_id = "backend_demo_001"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/inference/predict" -Method POST -ContentType "application/json" -Body $body
```

Expected result:

```text
status: success
decision_type: fine_label
final_label: plastic
```

## 10. Run the Frontend Demo

Keep the backend running.

Open the frontend:

```powershell
start frontend\index.html
```

Use the default values:

```text
Image Path: ml/data/local_unknown/images/local_000001.jpg
Sample ID: local_000001
Backend URL: http://127.0.0.1:8000
```

Click:

```text
Run Prediction
```

Expected browser result:

```text
Prediction completed successfully.
Final label: plastic
Decision type: fine_label
Confidence: 0.962933
```

## 11. Stop the Backend

In the backend terminal:

```text
CTRL + C
```

## Troubleshooting

| Problem                              | Likely Cause                | Fix                                                                           |                                 |
| ------------------------------------ | --------------------------- | ----------------------------------------------------------------------------- | ------------------------------- |
| ModuleNotFoundError for openwaste_hr | PYTHONPATH not set          | run `$env:PYTHONPATH="ml/src;."`                                              |                                 |
| Checkpoint not found                 | wrong checkpoint path       | check `ml/outputs/checkpoints/baseline_trashnet_v1/baseline_trashnet_best.pt` |                                 |
| Frontend request fails               | backend not running         | start Uvicorn                                                                 |                                 |
| CORS error in browser                | backend CORS not active     | check `backend/app/main.py` has `CORSMiddleware`                              |                                 |
| Prediction endpoint returns error    | image path invalid          | use a project-relative image path                                             |                                 |
| PowerShell multiline issue           | command not closed properly | retype the `$body = @{ ... }                                                  | ConvertTo-Json` block carefully |

## Current Demo Image Result

For:

```text
ml/data/local_unknown/images/local_000001.jpg
```

Current expected output:

| Field                        | Value                |
| ---------------------------- | -------------------- |
| pred_label                   | plastic              |
| max_softmax_confidence       | 0.962933             |
| top_coarse_label             | recyclable           |
| top_coarse_confidence        | 0.999999             |
| hierarchical_decision_type   | fine_label           |
| hierarchical_final_label     | plastic              |
| hierarchical_decision_reason | fine_confidence_high |

## OpenWaste-HR Decision Types

The prototype can return three final decision types:

| Decision Type | Meaning                                                                                   |
| ------------- | ----------------------------------------------------------------------------------------- |
| fine_label    | the system returns a detailed known waste label                                           |
| coarse_label  | the system returns a safer broad fallback category                                        |
| manual_review | the system routes the image to human review because the prediction is uncertain or unsafe |

## Research Meaning

This guide shows that OpenWaste-HR is now a working prototype workflow.

The system can move from an image path to a safe hierarchical decision through command-line inference, a backend endpoint, and a simple browser-based frontend.
