# Methodology Chapter Consolidated Draft v1

## 1. Introduction

This chapter describes the methodology used to design, implement, and evaluate OpenWaste-HR, a hierarchical open-set waste classification prototype with reject/manual-review decisions and local active learning support.

The methodology was designed around the limitation of conventional closed-set waste classifiers. A normal image classifier always assigns an input image to one of the known training classes, even when the object is ambiguous, mixed, unfamiliar, or outside the dataset distribution. In real-world waste classification, this behaviour can be unsafe because the system may confidently assign an incorrect known label to an unknown or locally specific waste item.

To address this problem, OpenWaste-HR uses a methodology based on:

* hierarchical fine-to-coarse waste taxonomy
* known-class baseline training
* reject-option evaluation
* local unknown evaluation
* hierarchical decision-making
* safe threshold tuning
* active learning candidate selection
* human labelling workflow
* prototype-level inference, backend, and frontend validation

The methodology therefore evaluates the system not only by accuracy, but also by its ability to handle uncertainty and route difficult samples to manual_review.

## 2. Research Design

The project follows an experimental prototype methodology.

The research process was organised into the following stages:

```text id="y131e7"
taxonomy design → dataset preparation → baseline model training → closed-set evaluation → reject-option evaluation → local unknown evaluation → hierarchical decision policy → safe policy tuning → active learning workflow → inference pipeline → backend/frontend prototype
```

Each stage produced implementation artefacts, metrics, reports, and tests. This allowed the project to develop from a basic classifier into a complete prototype workflow.

The methodology was designed to answer the following practical research questions:

| Question                                                    | Methodological Response                                                |
| ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| Can a baseline model classify known waste images?           | train and evaluate a closed-set classifier                             |
| Can uncertain predictions be rejected?                      | apply confidence, max-logit, and energy reject baselines               |
| How does the system behave on local unknown images?         | evaluate rejection and false acceptance on local unknown data          |
| Can a safer fallback be produced instead of only rejecting? | implement fine_label, coarse_label, and manual_review decisions        |
| Can local unknown samples be used for improvement?          | select active learning candidates and prepare human labelling workflow |
| Can the prototype be demonstrated end-to-end?               | implement inference, backend endpoint, and frontend demo               |

## 3. Taxonomy Methodology

OpenWaste-HR uses a hierarchical taxonomy with fine labels and coarse labels.

The current trained model includes the following fine-to-coarse mapping:

| Fine Label      | Coarse Label |
| --------------- | ------------ |
| paper_cardboard | recyclable   |
| plastic         | recyclable   |
| glass           | recyclable   |
| metal           | recyclable   |
| residual        | residual     |

The broader target taxonomy also reserves space for categories such as organic and e_waste_battery. These are important for future dataset expansion, but are not fully represented in the current trained baseline.

The methodology also defines two special concepts:

| Concept       | Meaning                                                                                              |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| unknown       | an input image that does not clearly belong to the known training classes                            |
| manual_review | a decision outcome where the system avoids automatic acceptance and routes the image to human review |

This taxonomy supports the key methodological idea: the system should not be forced to return only fine labels. It should be able to return a fine_label, a safer coarse_label, or a manual_review decision.

## 4. Dataset Methodology

The dataset methodology uses a manifest-based design. Instead of relying only on raw folder names, each image is represented in a structured CSV manifest.

The manifest records important information such as:

| Field          | Purpose                                                         |
| -------------- | --------------------------------------------------------------- |
| sample_id      | unique image identifier                                         |
| source_dataset | dataset origin                                                  |
| image_path     | project-relative image path                                     |
| original_label | original dataset label                                          |
| fine_label     | mapped OpenWaste-HR fine label                                  |
| coarse_label   | mapped OpenWaste-HR coarse label                                |
| is_known       | whether the image belongs to a known class                      |
| usage          | train, validation, test, unknown_test, or active learning usage |

This makes the dataset pipeline easier to audit, test, and extend.

The first known dataset was prepared from a TrashNet-style folder structure. The generated split sizes were:

| Split      | Samples |
| ---------- | ------: |
| Train      |    1766 |
| Validation |     377 |
| Test       |     384 |
| Total      |    2527 |

A separate local unknown dataset was prepared for open-set evaluation. These images were not treated as normal training classes. Instead, they were used to test whether the system could avoid unsafe acceptance of images outside the known dataset.

## 5. Dataset Inspection Methodology

A dataset inspection stage was used before training. This checked:

| Inspection Area     | Purpose                                 |
| ------------------- | --------------------------------------- |
| image readability   | detect missing or corrupted image files |
| label distribution  | identify imbalance between classes      |
| image dimensions    | understand input consistency            |
| source labels       | verify original-to-taxonomy mapping     |
| known/unknown usage | confirm correct split assignment        |

The inspection stage confirmed that the baseline dataset was usable, but also revealed limited taxonomy coverage and class imbalance. This justified the later use of reject options, local unknown testing, and active learning.

## 6. Baseline Training Methodology

The first baseline model was implemented as a closed-set image classifier.

The model pipeline included:

| Component           | Method                                             |
| ------------------- | -------------------------------------------------- |
| image preprocessing | resize and tensor conversion                       |
| dataset loading     | PyTorch dataset from manifest rows                 |
| label encoding      | fine labels mapped to class indices                |
| model               | MobileNetV3-style classifier                       |
| optimiser           | AdamW                                              |
| validation metric   | macro-F1                                           |
| checkpointing       | best checkpoint and final checkpoint               |
| evaluation          | accuracy, balanced accuracy, macro-F1, weighted-F1 |

This first baseline was trained from scratch and used as the original comparison point. Later methodology stages will train pretrained and dataset-expanded models for comparison.

## 7. Closed-Set Evaluation Methodology

The closed-set evaluation measured the baseline classifier under the standard assumption that every test image belongs to one of the known classes.

The metrics used were:

| Metric            | Purpose                                  |
| ----------------- | ---------------------------------------- |
| accuracy          | overall classification correctness       |
| balanced accuracy | class-balanced correctness               |
| macro-F1          | equal-weighted average F1 across classes |
| weighted-F1       | class-frequency-weighted F1              |

Closed-set evaluation is necessary because it provides the first baseline. However, it is not sufficient for OpenWaste-HR because real waste images can be unknown, mixed, or ambiguous.

## 8. Reject-Option Methodology

Reject-option evaluation was added to reduce unsafe forced predictions.

Three reject scoring methods were evaluated:

| Method               | Score                         |
| -------------------- | ----------------------------- |
| confidence threshold | maximum softmax confidence    |
| max-logit score      | maximum raw model logit       |
| energy score         | energy-based confidence score |

Each method used validation data to select an acceptance threshold. A sample was accepted only if the score satisfied the selected threshold; otherwise, it was routed to manual_review.

The reject-option methodology used metrics such as:

| Metric             | Purpose                                      |
| ------------------ | -------------------------------------------- |
| coverage           | percentage of samples accepted automatically |
| rejection rate     | percentage of samples routed to review       |
| selective accuracy | accuracy among accepted samples only         |
| selective macro-F1 | macro-F1 among accepted samples only         |

This stage tested whether accepted predictions became more reliable after rejecting uncertain samples.

## 9. Local Unknown Evaluation Methodology

The local unknown evaluation tested whether the system could avoid accepting images outside the known training distribution.

The local unknown dataset was used only for evaluation and active learning workflow development. It was not treated as an ordinary known class.

The main metrics were:

| Metric                              | Meaning                                                          |
| ----------------------------------- | ---------------------------------------------------------------- |
| unknown rejection rate              | proportion of local unknown samples rejected or routed to review |
| unknown false acceptance rate       | proportion of local unknown samples accepted as known            |
| accepted unknown label distribution | which known labels unknown samples were incorrectly accepted as  |

This methodology is important because open-world waste classification cannot be evaluated using only known-test accuracy.

## 10. Hierarchical Decision Methodology

The hierarchical decision policy was implemented to provide a safer alternative to simple fine-label prediction.

The policy can output:

| Decision Type | Meaning                            |
| ------------- | ---------------------------------- |
| fine_label    | return a detailed known fine label |
| coarse_label  | return a broader fallback category |
| manual_review | route the image to human review    |

The policy uses fine-label confidence, coarse-level confidence, and coarse-level margin to decide whether an image should receive a fine label, coarse label, or manual review.

The methodological advantage is that the system can still provide useful information when fine-level confidence is not safe, while avoiding automatic acceptance when uncertainty is too high.

## 11. Safe Policy Tuning Methodology

The first hierarchical policy achieved high known-test coverage but accepted too many local unknown images. Therefore, safe policy tuning was added.

The tuning process searched combinations of:

| Threshold                     | Purpose                                          |
| ----------------------------- | ------------------------------------------------ |
| fine_confidence_threshold     | minimum confidence for fine-label acceptance     |
| coarse_confidence_threshold   | minimum confidence for coarse fallback           |
| coarse_margin_threshold       | minimum separation between top coarse groups     |
| minimum_confidence_for_coarse | minimum evidence required before coarse fallback |

The selected safe policy was chosen to improve the balance between known-test reliability and local unknown manual-review behaviour.

The safe policy selected:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.900000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.650000 |
| minimum_confidence_for_coarse | 0.650000 |

This methodology explicitly treats safety and uncertainty as part of the evaluation process.

## 12. Active Learning Methodology

Active learning was included to support local improvement over time.

The active learning workflow is:

```text id="3fn7v2"
safe policy decisions → candidate scoring → candidate selection → human labelling sheet → reviewed label processing → future dataset v2
```

The candidate selection stage prioritises samples that may be useful for improving the system, including:

| Candidate Type                    | Reason                                                       |
| --------------------------------- | ------------------------------------------------------------ |
| manual_review samples             | uncertain or unsafe samples                                  |
| coarse_label samples              | cases where broad category is safer than fine label          |
| low-confidence fine_label samples | potentially risky accepted predictions                       |
| high-risk local unknown samples   | unknown images that the model may confuse with known classes |

The first candidate batch selected 20 local samples for human review.

The human labelling sheet supports future annotation, and reviewed-label processing prepares human feedback for dataset updates.

## 13. Inference Methodology

Inference was implemented at multiple levels to support testing and demonstration.

| Inference Component    | Purpose                                      |
| ---------------------- | -------------------------------------------- |
| single-image inference | run the model and policy on one image        |
| batch inference        | process a folder of images                   |
| prototype API wrapper  | convert inference output into stable JSON    |
| backend endpoint       | expose inference through FastAPI             |
| frontend demo          | display prediction and decision in a browser |

The inference output includes:

| Output              | Meaning                                    |
| ------------------- | ------------------------------------------ |
| predicted label     | top fine-label prediction                  |
| confidence          | model confidence                           |
| top coarse label    | aggregated coarse-level prediction         |
| decision type       | fine_label, coarse_label, or manual_review |
| final label         | final system output                        |
| class probabilities | probability for each known fine label      |
| policy thresholds   | thresholds used to make the final decision |

This methodology ensures that the system can be evaluated both as an offline ML pipeline and as a usable prototype.

## 14. Backend and Frontend Prototype Methodology

A FastAPI backend was implemented to expose inference through an HTTP endpoint.

Current backend endpoints:

| Method | Endpoint               | Purpose                                 |
| ------ | ---------------------- | --------------------------------------- |
| GET    | /health                | check whether the backend is running    |
| POST   | /api/inference/predict | run OpenWaste-HR inference on one image |

A lightweight frontend was implemented using HTML, CSS, and JavaScript. The frontend sends an image path and sample ID to the backend and displays the returned hierarchical decision.

This prototype methodology supports demonstration, supervisor feedback, and future user-interface development.

## 15. Testing and Reproducibility Methodology

Automated tests were added throughout the project.

The tests cover:

| Area                | Examples                                                 |
| ------------------- | -------------------------------------------------------- |
| taxonomy            | fine/coarse mappings and reserved labels                 |
| manifest validation | required columns and usage values                        |
| dataset processing  | image loading and inspection                             |
| model utilities     | label encoding and model output shape                    |
| evaluation metrics  | classification, selective, unknown, and open-set metrics |
| hierarchical policy | decision logic and tuning                                |
| active learning     | candidate scoring and selection                          |
| human labelling     | annotation sheet and reviewed-label processing           |
| inference           | single-image and batch inference                         |
| API wrapper         | request and response format                              |
| backend             | health and prediction endpoints                          |
| frontend            | required UI files and API call logic                     |
| documentation       | run guide, summaries, architecture, and thesis drafts    |

This testing approach improves confidence that the prototype components behave as expected.

## 16. Methodological Limitations

The current methodology has several limitations:

| Limitation                        | Explanation                                                    |
| --------------------------------- | -------------------------------------------------------------- |
| limited trained taxonomy          | the current model uses five known fine labels                  |
| missing trained categories        | organic and e_waste_battery are not yet fully trained          |
| small local unknown set           | current local unknown evaluation uses 40 samples               |
| human review pending              | active learning labels have not yet been filled                |
| no active-learning retraining yet | reviewed labels must be added before retraining                |
| first baseline only               | pretrained and expanded-dataset models are still planned       |
| simple frontend                   | the current frontend uses image paths instead of direct upload |

These limitations are expected at the current prototype stage and will be addressed in later iterations.

## 17. Future Methodological Extensions

The next methodology stages will include:

| Extension                  | Purpose                                                |
| -------------------------- | ------------------------------------------------------ |
| pretrained model training  | compare transfer learning against the scratch baseline |
| additional public datasets | improve label coverage and dataset diversity           |
| human correction labels    | add reviewed local feedback                            |
| dataset v2 creation        | build an improved dataset from reviewed samples        |
| active-learning retraining | compare local-feedback model against baseline          |
| improved open-set scoring  | reduce unknown false acceptance                        |
| frontend image upload      | make the demo easier to use                            |

These future stages will allow the final evaluation chapter to compare the original baseline against stronger models and active-learning updates.

## 18. Conclusion

The OpenWaste-HR methodology combines standard supervised classification with uncertainty-aware decision-making and local active learning.

The project does not evaluate success only through closed-set accuracy. Instead, it evaluates how the system behaves when predictions are uncertain, when local unknown images are encountered, and when human feedback can be used for future improvement.

This methodology supports the central research contribution: hierarchical open-set waste classification with reject/manual-review decisions and local active learning support.
