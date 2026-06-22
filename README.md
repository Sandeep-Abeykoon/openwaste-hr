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
* public unknown/future-class evaluation
* safe threshold tuning
* backend/frontend prototype integration
* human-in-the-loop active learning preparation

The main research argument is that waste classification should not be treated only as a closed-set image classification problem. In real-world use, local waste images may contain unfamiliar objects outside the training taxonomy. A normal classifier may still produce a high-confidence known label for these unknown objects.

OpenWaste-HR addresses this by allowing the system to return:

| Decision Type | Meaning                                   |
| ------------- | ----------------------------------------- |
| fine_label    | confident known fine-label prediction     |
| coarse_label  | safer broad fallback category             |
| manual_review | uncertain or unsafe case routed to review |

## Final Expanded Public Result

OpenWaste-HR was extended beyond the original TrashNet-only workflow by adding RealWaste as a second public dataset. The final system was evaluated as a hierarchical, uncertainty-aware waste classification pipeline rather than a simple closed-set classifier.

The final strongest balanced system is:

```text
Baseline C + Safe Policy C
```

Where:

| Component     | Meaning                                                            |
| ------------- | ------------------------------------------------------------------ |
| Baseline C    | pretrained expanded public model                                   |
| Safe Policy C | expanded public safe hierarchical decision policy using Baseline C |

The final known-class model was trained on the expanded public dataset created from TrashNet-style known samples and RealWaste known samples.

The known fine labels are:

* paper/cardboard
* plastic
* glass
* metal
* organic
* residual

RealWaste `Textile Trash` was intentionally kept outside the known training taxonomy and used as public unknown/future-class evaluation data.

## Final Closed-Set Known Classification

| Model                            | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| -------------------------------- | -------: | ----------------: | -------: | ----------: |
| Scratch TrashNet-only baseline   |   0.6927 |            0.6545 |   0.6456 |      0.7009 |
| Pretrained TrashNet-only model   |   0.8880 |            0.8431 |   0.8510 |      0.8873 |
| Pretrained expanded public model |   0.8876 |            0.8750 |   0.8819 |      0.8870 |

The expanded public model achieved similar accuracy to the TrashNet-only pretrained model, but improved macro-F1. This matters because macro-F1 better reflects class-balanced performance. The expanded public model also includes the organic class, which was missing from the original TrashNet-only known training setup.

## Final Reject-Option Result

Reject-option evaluation tested whether the model could improve accepted prediction reliability by rejecting uncertain known-test samples.

| Method               | Coverage | Rejection Rate | Selective Macro-F1 | Selective Weighted-F1 |
| -------------------- | -------: | -------------: | -----------------: | --------------------: |
| Confidence threshold |   0.7229 |         0.2771 |             0.9732 |                0.9788 |
| Max-logit            |   0.7362 |         0.2638 |             0.9627 |                0.9676 |
| Energy score         |   0.7181 |         0.2819 |             0.9612 |                0.9668 |

Confidence thresholding was strongest for known selective prediction because it produced the highest selective macro-F1 and selective weighted-F1.

## Final Unknown Handling Result

Standalone unknown rejection showed that Energy score was strongest for unknown detection.

| Unknown Source                    | Best Method  | Unknown Rejection Rate | False Acceptance Rate |
| --------------------------------- | ------------ | ---------------------: | --------------------: |
| Local unknown dataset             | Energy score |                 0.6750 |                0.3250 |
| Public unknown/future-class split | Energy score |                 0.6509 |                0.3491 |

This shows an important research trade-off. Confidence thresholding was strongest for selective known prediction, while Energy score was strongest for unknown rejection.

## Final Safe Hierarchical Policy

The final expanded public safe hierarchical policy selected these thresholds:

| Parameter                              | Value |
| -------------------------------------- | ----: |
| fine confidence threshold              | 0.995 |
| coarse confidence threshold            | 0.990 |
| coarse margin threshold                | 0.150 |
| minimum confidence for coarse fallback | 0.350 |

Final safe policy result:

| Policy                      | Known Coverage | Accepted Success Rate | Local Unknown Manual Review Rate |
| --------------------------- | -------------: | --------------------: | -------------------------------: |
| TrashNet-only safe policy   |         0.8646 |                0.9608 |                           0.6000 |
| Expanded public safe policy |         0.8819 |                0.9838 |                           0.4750 |

The expanded public safe policy achieved higher known coverage and higher accepted-decision reliability. The earlier TrashNet-only safe policy was stricter on local unknown samples. Therefore, the final result should be presented as a trade-off, not as one system being better in every metric.

## Final Research Position

The final thesis position is that OpenWaste-HR should not be treated as a simple closed-set waste classifier.

The final system supports:

| Output          | Meaning                               |
| --------------- | ------------------------------------- |
| fine label      | high-confidence fine waste prediction |
| coarse fallback | safer higher-level waste category     |
| manual review   | uncertain or unknown-like sample      |

The final expanded public safe hierarchical policy is the strongest balanced OpenWaste-HR system. However, the thesis should also report that a future version could add an energy-based unknown safety gate to improve unknown rejection further.

## Human Review and Active Learning Status

OpenWaste-HR includes a local active learning workflow, but the full human-review retraining cycle has not yet been completed.

Completed active learning components:

| Component                                    | Status   |
| -------------------------------------------- | -------- |
| active learning candidate selection          | complete |
| human labelling sheet generation             | complete |
| reviewed-label processing script             | complete |
| reviewed local label seed for `local_000001` | complete |
| active learning v2 dataset planning          | complete |

Not yet completed:

| Component                                           | Status  |
| --------------------------------------------------- | ------- |
| full manual review of remaining candidate images    | pending |
| retraining with reviewed known-class samples        | pending |
| before/after active learning improvement comparison | pending |

The reviewed local example `local_000001` was identified as a rubber slipper / flip-flop, which is outside the current known taxonomy. It was therefore kept as an unknown/future-class candidate rather than being added incorrectly into a known class.

This is important because active learning should improve the model only when reviewed samples genuinely belong to the current known taxonomy. Unknown or outside-taxonomy samples should be used for unknown evaluation, future-class planning, or manual-review analysis.

## Local Unknown Example

The live demo uses:

```text
ml/data/local_unknown/images/local_000001.jpg
```

Human observation:

```text
rubber slipper / flip-flop
```

This object is outside the current known fine-label taxonomy.

This example demonstrates why OpenWaste-HR needs local unknown evaluation and manual-review support. A normal closed-set classifier may still assign a known label to an unfamiliar object.

The earlier model/API result for this sample was:

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
330 passed, 1 warning
```

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

## Key Thesis and Result Files

| Purpose                               | File                                                             |
| ------------------------------------- | ---------------------------------------------------------------- |
| methodology chapter                   | docs/thesis/methodology_chapter_consolidated_v1.md               |
| implementation chapter                | docs/thesis/implementation_chapter_draft_v1.md                   |
| evaluation chapter draft              | docs/thesis/evaluation_chapter_draft_v1.md                       |
| final expanded public comparison      | docs/results/final_expanded_public_model_policy_comparison_v1.md |
| final thesis evaluation update        | docs/thesis/evaluation_expanded_public_final_update_v1.md        |
| final project summary after expansion | docs/thesis/final_project_summary_after_expansion_v1.md          |
| active learning v2 section            | docs/thesis/active_learning_v2_section_v1.md                     |
| thesis assembly checklist             | docs/thesis/thesis_assembly_checklist_v1.md                      |
| supervisor handover pack              | docs/supervisor_updates/supervisor_handover_pack_v1.md           |
| supervisor demo script                | docs/supervisor_updates/supervisor_demo_script_v1.md             |
| final supervisor completion summary   | docs/supervisor_updates/final_project_completion_summary_v1.md   |

## Completed Pipeline

| Stage                                                  | Status   |
| ------------------------------------------------------ | -------- |
| taxonomy and label map                                 | complete |
| dataset manifest validation                            | complete |
| TrashNet intake                                        | complete |
| scratch baseline training                              | complete |
| confidence/max-logit/energy reject-option baselines    | complete |
| local unknown dataset evaluation                       | complete |
| hierarchical decision policy                           | complete |
| safe policy tuning                                     | complete |
| pretrained TrashNet training                           | complete |
| pretrained reject/local unknown evaluation             | complete |
| pretrained safe policy selection                       | complete |
| RealWaste intake                                       | complete |
| RealWaste inspection                                   | complete |
| expanded public manifest creation                      | complete |
| expanded public pretrained training                    | complete |
| expanded public closed-set evaluation                  | complete |
| expanded public reject-option evaluation               | complete |
| expanded public local unknown evaluation               | complete |
| expanded public public unknown/future-class evaluation | complete |
| expanded public safe hierarchical policy tuning        | complete |
| final expanded public comparison                       | complete |
| backend/frontend prototype                             | complete |
| human correction seed                                  | complete |
| active learning v2 dataset plan                        | complete |
| thesis section updates                                 | complete |
| supervisor handover pack                               | complete |

## Remaining Work

Recommended future work:

1. complete manual review for the remaining active-learning candidate images
2. identify which reviewed samples truly belong to the current known taxonomy
3. build an active learning v2 retraining manifest using only valid reviewed known-class samples
4. fine-tune the expanded public pretrained model using reviewed known-class samples
5. compare before/after active learning performance
6. evaluate TACO as the next public dataset
7. decide whether TACO should be used for training, unknown evaluation, or future-class analysis
8. evaluate more pretrained architectures
9. improve frontend upload support
10. prepare final dissertation formatting and citations
11. deploy the backend/frontend prototype

## Next Dataset Work

The next planned dataset stage is TACO feasibility and intake planning.

TACO should not be added blindly into training. It requires a careful label-mapping protocol because its labels are more detailed and may not directly fit the current six known fine labels.

The next dataset decision should classify TACO labels into:

| Mapping Role           | Meaning                                             |
| ---------------------- | --------------------------------------------------- |
| known_train_candidate  | can safely map to a current known fine label        |
| known_eval_candidate   | can be evaluated as a known class                   |
| unknown_eval_candidate | useful as unknown/outside-taxonomy evaluation       |
| future_class_candidate | useful for future taxonomy expansion                |
| exclude_or_review      | too ambiguous or unsuitable without manual checking |

## Terminology

Use these terms consistently:

| Use This Term                            | Avoid                   |
| ---------------------------------------- | ----------------------- |
| local unknown dataset                    | unclear unknown dataset |
| manual_review                            | reject only             |
| coarse fallback                          | wrong broad prediction  |
| expanded public safe hierarchical policy | final model only        |
| human-in-the-loop active learning        | manual checking only    |
| outside_current_known_taxonomy           | wrong label             |

## Current Status

OpenWaste-HR now has a complete expanded public research and prototype pipeline.

The strongest current thesis message is:

```text
OpenWaste-HR improves waste classification by combining pretrained image recognition with hierarchical open-set decision-making, safe reject/manual-review behaviour, local unknown evaluation, public unknown evaluation, and human-in-the-loop active learning preparation.
```

The next major research step is to complete the human-review retraining cycle and then continue with TACO dataset feasibility and intake planning.
