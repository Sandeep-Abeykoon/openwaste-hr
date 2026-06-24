# OpenWaste-HR

## Project Title

**OpenWaste-HR: Safety-Aware Open-Set Waste Classification with Active Learning and Fusion-Gate Manual Review**

## Project Summary

OpenWaste-HR is a final-year research and prototype project for safer real-world waste classification.

The project investigates why waste classification should not be treated only as a closed-set image classification problem. In real-world use, a waste image may contain an item outside the model's known training taxonomy. A normal classifier may still force such an unknown item into one of the known classes with high confidence.

OpenWaste-HR addresses this by combining:

- staged multi-source dataset expansion,
- pretrained image classification,
- active-learning-supported dataset adaptation,
- temperature-scaled confidence calibration,
- reject-option open-set evaluation,
- Mahalanobis feature-distance scoring,
- a calibrated Fusion Gate v2 accept/reject policy,
- final manual-review routing for unknown or unsafe inputs,
- backend/frontend prototype integration for prediction demonstration.

The final system does not simply return a class label for every image. Instead, it decides whether the image is safe to classify as a known recyclable item or whether it should be sent to manual review.

---

## Final Research Position

The final thesis position is:

```text
OpenWaste-HR shows that a closed-set waste classifier becomes safer for real-world deployment when combined with a validation-calibrated reject-option layer. The final Mahalanobis-enhanced Fusion Gate v2 reduced false acceptance of unknown waste from 0.1500 to 0.0663 while preserving similar known-class coverage, demonstrating the value of combining score-level uncertainty and feature-space evidence.
```

The final selected decision policy is:

```text
Stage 4 MobileNetV3 classifier
+ temperature-scaled confidence display
+ Mahalanobis feature-distance knownness
+ Fusion Gate v2 knownness threshold
+ manual review for unsafe/unknown inputs
```

---

## Final Known Taxonomy

The final classifier is trained on five known recyclable fine classes:

| Known class |
|---|
| cardboard |
| glass |
| metal |
| paper |
| plastic |

These known classes map to the coarse category:

```text
recyclable
```

The project uses a safety-aware coarse-to-fine decision policy:

```text
accepted known fine label → recyclable
unknown or unsafe input → manual review
```

This is not claimed as a rich multi-level waste hierarchy. It is a practical safety-aware mapping from supported known fine labels to a recyclable coarse category, with unknown items routed to manual review.

---

## Unknown / Open-Set Evaluation Classes

The final unknown classes used for validation and testing are:

| Unknown class |
|---|
| biological |
| textile |

These unknown classes are not trained as a sixth class. They are used only for:

- validation threshold selection,
- reject-option evaluation,
- final open-set testing,
- manual-review behaviour analysis.

---

## Final Dataset Sources

The final Stage 4 known-class dataset combines cleaned known-class samples from:

| Dataset |
|---|
| TrashNet |
| RealWaste |
| Garbage Classification V2 |
| TrashBox |

The final unknown validation/test data comes from held-out biological and textile samples.

---

## Final Dataset Split Summary

| Split | Rows |
|---|---:|
| known_train | 15,958 |
| known_val | 3,417 |
| known_test | 3,426 |
| unknown_val | 1,660 |
| unknown_test | 1,660 |

---

## Staged Closed-Set Results

| Stage | Description | Test Accuracy | Test Macro-F1 | Test Balanced Accuracy |
|---|---|---:|---:|---:|
| Stage 1 | TrashNet baseline | 0.9324 | 0.9320 | 0.9357 |
| Stage 2 | TrashNet + RealWaste | 0.9432 | 0.9447 | 0.9445 |
| Stage 3 | Add Garbage V2 | 0.9445 | 0.9445 | 0.9437 |
| Stage 3 + Active Learning | Add reviewed active-learning samples | 0.9508 | 0.9509 | 0.9510 |
| Stage 4 | Add cleaned TrashBox | 0.9212 | 0.9213 | 0.9217 |

The Stage 4 model is selected as the final known-class classifier because it uses the broadest cleaned multi-source dataset.

---

## Final Stage 4 Per-Dataset Results

| Dataset | Rows | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|
| TrashNet | 355 | 0.9634 | 0.9636 | 0.9639 |
| RealWaste | 472 | 0.9089 | 0.9111 | 0.9101 |
| Garbage V2 | 1,082 | 0.9538 | 0.9532 | 0.9551 |
| TrashBox | 1,517 | 0.8919 | 0.8916 | 0.8915 |
| Overall | 3,426 | 0.9212 | 0.9213 | 0.9217 |

---

## Active Learning Findings

Active learning was used between staged dataset expansions to select uncertain or informative external samples for human review.

| Comparison | External Dataset | Baseline Macro-F1 | Active Learning Macro-F1 | Result |
|---|---|---:|---:|---|
| Stage 1 to Stage 2 | RealWaste | 0.5161 | 0.7606 | Improved |
| Stage 2 to Stage 3 | Garbage V2 | 0.7494 | 0.8540 | Improved |
| Stage 3 to Stage 4 | TrashBox | 0.6292 | 0.6901 | Improved |

The results show that active learning improved adaptation to new dataset sources before full staged expansion.

---

## Calibration Result

Temperature scaling was applied to improve confidence calibration.

| Metric | Before Temperature Scaling | After Temperature Scaling |
|---|---:|---:|
| Validation ECE | 0.0397 | 0.0118 |
| Test ECE | 0.0430 | 0.0084 |

The selected temperature was:

```text
1.8482154607772827
```

Temperature-scaled confidence is used for confidence display. The final accept/reject decision is made by Fusion Gate v2.

---

## Reject-Option and Fusion-Gate Results

| Method | Test AUROC | Known Coverage | Unknown Rejection | FAR | Accepted-Known Accuracy |
|---|---:|---:|---:|---:|---:|
| Confidence threshold | 0.8498 | 0.6842 | 0.8831 | 0.1169 | 0.9906 |
| Temperature-scaled confidence | 0.8572 | 0.7341 | 0.8530 | 0.1470 | 0.9877 |
| Max-logit score | 0.8782 | 0.7659 | 0.8506 | 0.1494 | 0.9802 |
| Energy score | 0.8789 | 0.7665 | 0.8500 | 0.1500 | 0.9791 |
| Fusion Gate v1 score-only | 0.8793 | 0.7627 | 0.8536 | 0.1464 | 0.9790 |
| Mahalanobis-only | 0.5636 | 0.7607 | 0.3012 | 0.6988 | 0.9068 |
| Fusion Gate v2 + Mahalanobis | 0.9269 | 0.7656 | 0.9337 | 0.0663 | 0.9752 |

---

## Final Selected Reject-Option Policy

The final selected policy is:

```text
Fusion Gate v2 with Mahalanobis feature-distance
```

Final threshold:

```text
0.6314586412215439
```

Decision rule:

```text
if fusion_knownness_score >= 0.6314586412215439:
    accept predicted known class
else:
    send to manual review
```

---

## Improvement Over Energy-Only Baseline

| Metric | Energy-Only | Fusion Gate v2 + Mahalanobis | Change |
|---|---:|---:|---:|
| AUROC | 0.8789 | 0.9269 | +0.0480 |
| Known coverage | 0.7665 | 0.7656 | -0.0009 |
| Unknown rejection | 0.8500 | 0.9337 | +0.0837 |
| FAR | 0.1500 | 0.0663 | -0.0837 |
| Accepted-known accuracy | 0.9791 | 0.9752 | -0.0039 |

The final Fusion Gate v2 policy substantially reduced false acceptance of unknown samples while keeping known coverage almost unchanged.

---

## Statistical Evaluation of Final Fusion Gate v2

To strengthen the final evaluation, Fusion Gate v2 was also assessed using bootstrap confidence intervals, calibration metrics, and partial AUROC in low false-acceptance-rate regions.

| Metric | Point Estimate | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|
| Known coverage | 0.7656 | 0.7522 | 0.7799 |
| Unknown rejection rate | 0.9337 | 0.9217 | 0.9452 |
| False acceptance rate | 0.0663 | 0.0548 | 0.0783 |
| Accepted-known accuracy | 0.9752 | 0.9694 | 0.9809 |
| AUROC known vs unknown | 0.9269 | 0.9197 | 0.9347 |

Fusion Gate v2 calibration and low-FAR performance were also measured:

| Metric | Value |
|---|---:|
| Fusion Gate v2 ECE | 0.0641 |
| Fusion Gate v2 Brier score | 0.1087 |
| Standardized pAUC, FAR <= 0.05 | 0.8062 |
| Standardized pAUC, FAR <= 0.10 | 0.8421 |

These results show that the final Fusion Gate v2 improvement is supported by statistical uncertainty estimates, measured calibration quality, and additional low-FAR analysis.

---

## Final Inference Behaviour

### Known Plastic Example

| Field | Value |
|---|---|
| Image | ml/data/raw/trashbox/plastic/plastic 1777.jpg |
| Internal prediction | plastic |
| Fusion knownness score | 0.9969 |
| Threshold | 0.6315 |
| Decision | known_fine_label |
| User-visible label | plastic |
| Coarse label | recyclable |

User-facing message:

```text
This item is likely plastic. It belongs to the recyclable category.
```

### Unknown Textile Example

| Field | Value |
|---|---|
| Image | ml/data/raw/garbage_v2/clothes/clothes_319.jpg |
| Actual unknown type | textile / clothes |
| Internal prediction | paper |
| Fusion knownness score | 0.0713 |
| Threshold | 0.6315 |
| Decision | unknown_manual_review |
| User-visible label | manual_review_required |
| Internal prediction shown to user | false |

User-facing message:

```text
The system is not confident that this item belongs to the supported recyclable classes. Please send it for manual review.
```

---

## Deployment Readiness

The final model was also tested for inference efficiency.

| Runtime | Mean Latency |
|---|---:|
| PyTorch CPU | 20.07 ms |
| PyTorch CUDA | 8.38 ms |
| ONNX Runtime CPU | 4.95 ms |

This supports the feasibility of using the model in a prototype prediction interface.

---

## Repository Structure

```text
ml/
  api/                         FastAPI prediction API for final Fusion Gate v2 policy
  configs/                     taxonomy, preprocessing, training, and final policy configs
  data/                        raw datasets, manifests, and split files
  outputs/                     model outputs, metrics, reports, and inference examples
  scripts/                     training, evaluation, fusion-gate, inference, and utility scripts

web/
  prediction-ui/               React + Vite prediction interface

docs/
  methodology/                 method notes and protocol files
  results/                     evaluation reports, figures, and final result summaries

tests/                         pytest validation tests
```

---

## Setup

Create and activate a Python virtual environment:

```powershell
cd "D:\Github Repositories\openwaste-hr"

python -m venv .venv
.\.venv\Scripts\activate
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

If using the prediction API, also install:

```powershell
python -m pip install fastapi uvicorn python-multipart
```

---

## Run Final Single-Image Inference

Known example:

```powershell
python ml\scripts\infer_with_fusion_gate_v2_policy.py `
  --image "ml\data\raw\trashbox\plastic\plastic 1777.jpg" `
  --output-json "ml\outputs\inference_examples\fusion_gate_v2_known_example_v1.json"
```

Unknown/manual-review example:

```powershell
python ml\scripts\infer_with_fusion_gate_v2_policy.py `
  --image "ml\data\raw\garbage_v2\clothes\clothes_319.jpg" `
  --output-json "ml\outputs\inference_examples\fusion_gate_v2_unknown_example_v1.json"
```

---

## Run Prediction API

Open PowerShell in the project root:

```powershell
cd "D:\Github Repositories\openwaste-hr"

python -m uvicorn ml.api.predict_api:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -Method GET
```

---

## Run React Prediction Interface

Open a second PowerShell window:

```powershell
cd "D:\Github Repositories\openwaste-hr\web\prediction-ui"

npm install
npm run dev
```

Optional: if the frontend should call a different API host, create
`web/prediction-ui/.env` from `web/prediction-ui/.env.example` and set:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Open:

```text
http://127.0.0.1:5173
```

Upload an image and run prediction.

---

## Key Result Files

| Purpose | File |
|---|---|
| Final result summary | docs/results/openwaste_hr_final_results_summary_v1.md |
| Fusion Gate v2 results | docs/results/fusion_gate_v2_mahalanobis_results_v1.md |
| Final decision policy v2 | docs/methodology/final_decision_policy_v2_fusion_gate.md |
| Final inference examples | docs/results/fusion_gate_v2_inference_examples_v1.md |
| Fusion Gate v2 statistical evaluation | docs/results/fusion_gate_v2_statistical_evaluation_v1.md |
| ONNX and latency benchmark | docs/results/onnx_latency_benchmark_results_v1.md |
| Taxonomy protocol | docs/methodology/taxonomy_protocol_v1.md |
| Reject-option protocol | docs/methodology/reject_option_evaluation_protocol_v1.md |
| Preprocessing protocol | docs/methodology/preprocessing_augmentation_protocol_v1.md |

---

## Completed Pipeline

| Stage | Status |
|---|---|
| taxonomy and label mapping protocol | complete |
| multi-source dataset manifest | complete |
| dataset split creation | complete |
| staged dataset expansion | complete |
| Stage 1 to Stage 4 classifier training | complete |
| active-learning candidate selection and evaluation | complete |
| temperature scaling | complete |
| confidence, max-logit, and energy reject-option evaluation | complete |
| Fusion Gate v1 score-level evaluation | complete |
| Mahalanobis feature-distance scoring | complete |
| Fusion Gate v2 Mahalanobis-enhanced evaluation | complete |
| Fusion Gate v2 statistical evaluation | complete |
| final decision policy v2 | complete |
| final single-image inference script | complete |
| final FastAPI prediction backend | complete |
| React prediction interface | complete |
| thesis/report packaging | in progress |

---

## Remaining Work

The main research implementation is complete. Remaining work is mainly final packaging:

1. create final report-ready methodology, implementation, and evaluation writeups,
2. run repository tests after installing pytest,
3. prepare final dissertation formatting and citations.

---

## Current Status

OpenWaste-HR now has a complete final research pipeline and a strong final result.

The strongest current thesis message is:

```text
OpenWaste-HR improves real-world waste classification safety by combining staged multi-source classifier training, active learning, calibrated confidence, open-set reject-option evaluation, and a Mahalanobis-enhanced Fusion Gate v2 policy that significantly reduces false acceptance of unknown waste items.
```
