# OpenWaste-HR Prototype Demo Checklist v1

## Demo Goal

Show that OpenWaste-HR is no longer only an offline model experiment.

The project now supports a working end-to-end prototype:

```text
image path → backend inference → hierarchical decision → frontend display
```

## Before the Demo

| Check                         | Status |
| ----------------------------- | ------ |
| Virtual environment activated | [ ]    |
| PYTHONPATH set to `ml/src;.`  | [ ]    |
| Tests pass                    | [ ]    |
| Model checkpoint exists       | [ ]    |
| Backend starts successfully   | [ ]    |
| Frontend opens in browser     | [ ]    |

## Test Command

```powershell
$env:PYTHONPATH="ml/src;."
pytest tests/test_taxonomy.py tests/test_manifest.py tests/test_trashnet_manifest_builder.py tests/test_image_dataset.py tests/test_dataset_inspection.py tests/test_label_encoding.py tests/test_torch_dataset.py tests/test_baseline_model.py tests/test_classification_metrics.py tests/test_selective_metrics.py tests/test_open_set_scores.py tests/test_local_unknown_manifest_builder.py tests/test_unknown_metrics.py tests/test_hierarchical_decision.py tests/test_hierarchical_policy_tuning.py tests/test_active_learning_selection.py tests/test_human_labelling_sheet.py tests/test_reviewed_label_processing.py tests/test_inference_pipeline.py tests/test_batch_inference_pipeline.py tests/test_api_wrapper.py tests/test_backend_inference_endpoint.py tests/test_frontend_demo_files.py
```

Expected:

```text
110 passed
```

## Start Backend

```powershell
$env:PYTHONPATH="ml/src;."
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Expected:

```text
Application startup complete.
```

## Open Frontend

```powershell
start frontend\index.html
```

## Demo Input

| Field       | Value                                         |
| ----------- | --------------------------------------------- |
| Image Path  | ml/data/local_unknown/images/local_000001.jpg |
| Sample ID   | local_000001                                  |
| Backend URL | http://127.0.0.1:8000                         |

## Expected Demo Output

| Field                  | Value                |
| ---------------------- | -------------------- |
| status                 | success              |
| pred_label             | plastic              |
| max_softmax_confidence | 0.962933             |
| top_coarse_label       | recyclable           |
| decision_type          | fine_label           |
| final_label            | plastic              |
| reason                 | fine_confidence_high |

## What to Explain to Supervisor

### 1. This is not just a normal classifier

A normal classifier always forces a known label.

OpenWaste-HR can return:

| Output        | Meaning                             |
| ------------- | ----------------------------------- |
| fine_label    | detailed known label                |
| coarse_label  | safer broad fallback                |
| manual_review | uncertain case sent to human review |

### 2. The project handles open-world uncertainty

The local unknown dataset and reject/manual-review workflow show that the system is designed for real-world images that may not match the training dataset.

### 3. Active learning is included

The system can select useful local unknown images for human labelling.

Current active learning candidate batch:

| Candidate Type            | Count |
| ------------------------- | ----: |
| manual-review candidates  |    12 |
| coarse-label candidates   |     4 |
| fine-label candidates     |     4 |
| total selected candidates |    20 |

### 4. The prototype is end-to-end

Current working flow:

```text
frontend → FastAPI backend → API wrapper → inference pipeline → hierarchical decision → frontend result
```

## Demo Script

1. Open the project folder.
2. Run the test suite.
3. Start the backend.
4. Open the frontend page.
5. Click Run Prediction.
6. Show the final decision card.
7. Show the class probability table.
8. Explain why the decision is fine_label, coarse_label, or manual_review.
9. Explain how manual-review and active learning support the project novelty.

## Demo Result Recording

| Item                    | Result     |
| ----------------------- | ---------- |
| Test suite result       | 110 passed |
| Backend health endpoint | success    |
| Frontend request        | success    |
| Final label             | plastic    |
| Decision type           | fine_label |
| Confidence              | 0.962933   |
| Demo status             | working    |

## Current Limitation to Mention

The current model is a first baseline trained mainly on TrashNet-derived known labels. The stronger contribution is not only raw classification accuracy, but the safer hierarchical decision workflow, reject option, manual-review path, and active learning process.

## Next Improvement

Future improvements can include:

* stronger pretrained model
* better open-set scoring
* more local data
* reviewed active learning labels
* improved frontend image upload instead of only image path input
