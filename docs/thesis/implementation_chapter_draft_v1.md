# Implementation Chapter Draft v1

## 1. Introduction

This chapter presents the implementation of OpenWaste-HR, a hierarchical open-set waste classification prototype designed to support safer waste image classification under uncertainty. Unlike a conventional closed-set image classifier, the proposed system does not always force an image into one of the known training categories. Instead, it can return a fine-grained waste label, fall back to a broader coarse category, or route uncertain samples to manual review.

The implementation follows the main aim of the project: to develop a waste classification workflow that is more suitable for open-world conditions, where images may contain unfamiliar objects, mixed materials, unclear items, or locally specific waste types not represented in the training dataset.

The implemented prototype includes dataset preparation, baseline model training, reject-option evaluation, hierarchical decision-making, local unknown evaluation, active learning support, human labelling workflow, inference pipelines, a FastAPI backend endpoint, and a simple frontend demonstration page.

## 2. System Overview

OpenWaste-HR was implemented as a modular project with separate components for machine learning, backend services, frontend demonstration, documentation, and testing.

The current workflow is:

```text
dataset preparation → model training → closed-set evaluation → reject-option evaluation → hierarchical decision policy → safe threshold tuning → local unknown evaluation → active learning candidate selection → inference pipeline → backend endpoint → frontend demo
```

The implementation is organised into the following main folders:

| Folder              | Purpose                                                                           |
| ------------------- | --------------------------------------------------------------------------------- |
| ml/src/openwaste_hr | machine learning source code                                                      |
| ml/configs          | YAML configuration files                                                          |
| ml/data             | raw, processed, split, and local unknown data                                     |
| ml/outputs          | generated metrics, figures, checkpoints, and reports                              |
| backend/app         | FastAPI backend endpoint                                                          |
| frontend            | simple browser-based prototype demo                                               |
| docs                | methodology notes, results reports, supervisor updates, and thesis draft material |
| tests               | automated tests for core implementation components                                |

This structure was chosen to keep the prototype reproducible and easier to extend during later development.

## 3. Taxonomy Design

The project uses a hierarchical taxonomy with fine labels and coarse labels.

The current trained model uses five known fine labels:

| Fine Label      | Coarse Label |
| --------------- | ------------ |
| paper_cardboard | recyclable   |
| plastic         | recyclable   |
| glass           | recyclable   |
| metal           | recyclable   |
| residual        | residual     |

The broader target taxonomy also reserves space for additional categories such as organic and e_waste_battery. However, these classes are not yet fully represented in the current trained dataset.

Two special decision concepts are also used:

| Special Concept | Meaning                                                                |
| --------------- | ---------------------------------------------------------------------- |
| unknown         | image does not belong clearly to the known classes                     |
| manual_review   | image should be reviewed by a human rather than automatically accepted |

This taxonomy supports the key project idea: the system can make a detailed prediction when confidence is high, return a broader category when only coarse-level evidence is reliable, or avoid unsafe prediction by sending the sample to manual review.

## 4. Dataset Preparation

The first dataset pipeline was built around a TrashNet-style folder structure. A manifest builder was implemented to convert folder-based images into structured CSV files. These files record image paths, source dataset information, original labels, mapped fine labels, coarse labels, and usage splits.

The generated dataset split was:

| Split      | Samples |
| ---------- | ------: |
| Train      |    1766 |
| Validation |     377 |
| Test       |     384 |
| Total      |    2527 |

The dataset inspection stage checked image readability, label distribution, image dimensions, and missing files. This confirmed that the dataset could be used for baseline training, while also showing an important limitation: the dataset is dominated by recyclable classes and does not cover all target taxonomy categories.

This limitation is important because it motivates the open-set and manual-review components of OpenWaste-HR.

## 5. Baseline Model Implementation

A MobileNetV3-style image classifier was implemented as the first baseline model. The model was trained as a closed-set classifier using the known fine labels.

The training pipeline included:

| Component                | Implementation                               |
| ------------------------ | -------------------------------------------- |
| image loading            | PyTorch dataset and transforms               |
| label encoding           | fine-label to index mapping                  |
| model                    | MobileNetV3-style classifier                 |
| optimisation             | AdamW optimiser                              |
| class imbalance handling | class weights                                |
| validation monitoring    | macro-F1                                     |
| checkpointing            | best and final model checkpoints             |
| testing                  | closed-set metrics and classification report |

The baseline classifier produced the following test performance:

| Metric            |    Value |
| ----------------- | -------: |
| Test accuracy     | 0.692708 |
| Balanced accuracy | 0.654500 |
| Macro-F1          | 0.645600 |
| Weighted-F1       | 0.700900 |

This baseline provides a useful starting point, but it is not sufficient for real-world waste classification because it always forces every image into a known label.

## 6. Reject-Option Baselines

To address forced prediction, reject-option baselines were implemented. These methods decide whether to accept a prediction or route it to manual review.

Three scoring approaches were evaluated:

| Method               | Score Used                    |
| -------------------- | ----------------------------- |
| confidence threshold | maximum softmax confidence    |
| max-logit score      | maximum raw logit             |
| energy score         | energy-based confidence score |

The confidence-threshold reject baseline selected a threshold of 0.640000 using validation data. On the known test set, it accepted 262 out of 384 samples and rejected 122 samples. Its selective accuracy on accepted samples was 0.770992.

On the local unknown dataset, the confidence reject baseline produced:

| Metric                        |    Value |
| ----------------------------- | -------: |
| Local unknown samples         |       40 |
| Rejected unknown samples      |       14 |
| Accepted unknown samples      |       26 |
| Unknown rejection rate        | 0.350000 |
| Unknown false acceptance rate | 0.650000 |

This showed that a simple reject option improves safety compared with forced classification, but it still accepts many local unknown images as known classes.

## 7. Hierarchical Decision Policy

The hierarchical decision policy was implemented to go beyond simple accept/reject behaviour.

Instead of only choosing between fine label and manual review, the system can return:

| Decision Type | Meaning                                 |
| ------------- | --------------------------------------- |
| fine_label    | return a detailed known fine label      |
| coarse_label  | return a broader safe fallback category |
| manual_review | send the image to human review          |

The first hierarchical policy used confidence and coarse-level aggregation to decide whether an image should receive a fine label, coarse fallback, or manual-review decision.

The first policy performed well on known-test coverage:

| Metric                             |    Value |
| ---------------------------------- | -------: |
| Known total samples                |      384 |
| Fine decisions                     |      262 |
| Coarse fallback decisions          |       96 |
| Manual-review decisions            |       26 |
| Known decision coverage            | 0.932292 |
| Hierarchical success over accepted | 0.824022 |

However, it was too permissive for local unknown images:

| Metric                  |    Value |
| ----------------------- | -------: |
| Local unknown samples   |       40 |
| Fine accepts            |       26 |
| Coarse accepts          |       11 |
| Manual-review count     |        3 |
| Manual-review rate      | 0.075000 |
| Unknown acceptance rate | 0.925000 |

This result showed that high known-test coverage alone is not enough. A safer policy must also reduce unsafe acceptance of unknown images.

## 8. Safe Hierarchical Policy Tuning

A safe hierarchical tuning stage was implemented to search for stricter decision thresholds. The selected policy uses:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.900000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.650000 |
| minimum_confidence_for_coarse | 0.650000 |

The safe policy reduced coverage on known samples, but improved accepted-decision reliability and local unknown manual-review routing.

Known-test result:

| Metric                             |    Value |
| ---------------------------------- | -------: |
| Known total samples                |      384 |
| Fine decisions                     |      145 |
| Coarse fallback decisions          |      108 |
| Manual-review decisions            |      131 |
| Known decision coverage            | 0.658854 |
| Hierarchical success over accepted | 0.889328 |

Local unknown result:

| Metric                  |    Value |
| ----------------------- | -------: |
| Local unknown samples   |       40 |
| Fine accepts            |       16 |
| Coarse accepts          |        9 |
| Manual-review count     |       15 |
| Manual-review rate      | 0.375000 |
| Unknown acceptance rate | 0.625000 |

The safe hierarchical policy was selected as the current OpenWaste-HR decision policy because it better supports uncertainty-aware classification.

## 9. Active Learning Workflow

The project also implements an active learning workflow for local unknown images.

The active learning stage uses the safe hierarchical decisions to select samples that should be reviewed by a human. Candidate selection prioritises manual-review cases, uncertain predictions, coarse fallback cases, and suspicious fine-label acceptances.

The selected candidate batch contained:

| Candidate Type            | Count |
| ------------------------- | ----: |
| Manual-review candidates  |    12 |
| Coarse-label candidates   |     4 |
| Fine-label candidates     |     4 |
| Total selected candidates |    20 |

A human labelling sheet was then generated for these 20 candidates. The sheet includes empty annotation columns such as human decision, human fine label, human coarse label, proposed new label, human confidence, notes, reviewer, and review date.

The reviewed-label processing pipeline was also implemented. Since the human sheet has not yet been filled, the current status is:

| Metric                 | Value |
| ---------------------- | ----: |
| Total rows             |    20 |
| Reviewed rows          |     0 |
| Pending review rows    |    20 |
| Ready for dataset rows |     0 |

This completes the first active learning feedback loop at the data-management level.

## 10. Inference Pipeline

A single-image inference pipeline was implemented to make the trained model usable on new images. The pipeline loads the trained checkpoint, applies image preprocessing, produces fine-label probabilities, applies the safe hierarchical policy, and writes JSON and Markdown outputs.

For the demo image:

```text
ml/data/local_unknown/images/local_000001.jpg
```

The result was:

| Field                  | Value                |
| ---------------------- | -------------------- |
| pred_label             | plastic              |
| max_softmax_confidence | 0.962933             |
| top_coarse_label       | recyclable           |
| top_coarse_confidence  | 0.999999             |
| coarse_margin          | 0.999999             |
| decision_type          | fine_label           |
| final_label            | plastic              |
| final_confidence       | 0.962933             |
| reason                 | fine_confidence_high |

A batch inference pipeline was also implemented to process folders of images and export structured decision results to CSV.

## 11. Prototype API Wrapper

The prototype API wrapper converts raw inference output into a backend-friendly response format.

The wrapper returns:

| Section             | Meaning                                |
| ------------------- | -------------------------------------- |
| status              | success or error                       |
| request             | image path and sample ID               |
| prediction          | model prediction and confidence values |
| decision            | final OpenWaste-HR decision            |
| class_probabilities | fine-label probability values          |
| policy              | safe hierarchical thresholds           |
| metadata            | device and pipeline version            |

This wrapper separates backend integration from model internals and provides a stable response structure for the next layer.

## 12. Backend Implementation

A FastAPI backend skeleton was implemented with two endpoints:

| Method | Endpoint               | Purpose                                 |
| ------ | ---------------------- | --------------------------------------- |
| GET    | /health                | backend health check                    |
| POST   | /api/inference/predict | run OpenWaste-HR inference on one image |

The backend receives a project-relative image path and optional sample ID, calls the prototype API wrapper, and returns the structured OpenWaste-HR response.

The tested backend result for the demo image was:

| Field            | Value      |
| ---------------- | ---------- |
| status           | success    |
| pred_label       | plastic    |
| decision_type    | fine_label |
| final_label      | plastic    |
| final_confidence | 0.962933   |
| device           | cuda       |

This confirms that the ML pipeline can be accessed through a backend endpoint.

## 13. Frontend Demo

A simple HTML, CSS, and JavaScript frontend was created to demonstrate the system in a browser.

The frontend allows the user to enter:

| Field       | Meaning                     |
| ----------- | --------------------------- |
| image path  | project-relative image path |
| sample ID   | optional identifier         |
| backend URL | FastAPI backend address     |

The frontend displays:

| Output              | Meaning                                    |
| ------------------- | ------------------------------------------ |
| final label         | final OpenWaste-HR decision                |
| decision type       | fine_label, coarse_label, or manual_review |
| decision reason     | explanation of the decision                |
| model prediction    | top fine-label prediction                  |
| confidence          | model confidence                           |
| class probabilities | probability for each known label           |
| raw response        | full backend JSON output                   |

The frontend successfully called the backend and displayed the expected result for `local_000001`.

## 14. Testing

Automated tests were created throughout the implementation. The current test suite covers:

| Area                | Examples                                                 |
| ------------------- | -------------------------------------------------------- |
| taxonomy            | fine/coarse label validation                             |
| manifest processing | required columns and split validation                    |
| dataset loading     | image dataset and inspection                             |
| model utilities     | label encoding and baseline model shape                  |
| evaluation metrics  | classification, selective, open-set, and unknown metrics |
| hierarchical policy | decision logic and threshold tuning                      |
| active learning     | candidate scoring and selection                          |
| human labelling     | labelling sheet and reviewed-label processing            |
| inference           | single-image and batch inference utilities               |
| API wrapper         | request validation and response format                   |
| backend             | health and prediction endpoints                          |
| frontend            | required HTML, CSS, and JS files                         |
| documentation       | prototype run guide and final summary reports            |

The latest test run passed 121 tests with one non-blocking FastAPI/TestClient warning.

## 15. Limitations

The current implementation has several limitations:

| Limitation                     | Explanation                                                    |
| ------------------------------ | -------------------------------------------------------------- |
| limited known classes          | the trained model currently uses five fine labels              |
| incomplete taxonomy coverage   | organic and e_waste_battery are not yet trained classes        |
| local unknown false acceptance | the safe policy still accepts some unknown images              |
| no completed human review yet  | active learning candidates have not yet been manually labelled |
| simple frontend                | the current frontend uses image paths rather than file upload  |
| baseline model only            | stronger pretrained models may improve performance             |

## 16. Conclusion

The OpenWaste-HR implementation demonstrates a working hierarchical open-set waste classification prototype.

The system is not limited to forced classification. It supports fine-label prediction, coarse fallback, manual-review routing, local unknown evaluation, active learning candidate selection, human labelling workflow, and an end-to-end backend/frontend demonstration.

This implementation provides a strong foundation for the thesis evaluation chapter and future improvements.
