# Evaluation Chapter Draft v1

## 1. Introduction

This chapter presents the current evaluation of OpenWaste-HR, a hierarchical open-set waste classification prototype with reject/manual-review decisions and local active learning support.

The evaluation does not only measure standard classification accuracy. Since the project focuses on open-world waste classification, the evaluation also considers whether the system can avoid unsafe forced predictions when it is uncertain or when an image may not belong to the known training classes.

The current evaluation is based on the first working prototype. It includes closed-set classification results, reject-option baselines, local unknown evaluation, hierarchical decision-policy evaluation, safe threshold tuning, active learning candidate selection, and prototype-level testing through command-line, backend, and frontend interfaces.

This version should be treated as the first evaluation chapter draft. Later versions can be expanded after adding pretrained training, additional datasets, filled human labels, and retraining comparisons.

## 2. Evaluation Objectives

The evaluation was designed around the following objectives:

| Objective                                | Purpose                                                               |
| ---------------------------------------- | --------------------------------------------------------------------- |
| Evaluate closed-set baseline performance | measure normal forced classification ability                          |
| Evaluate reject-option behaviour         | test whether uncertain samples can be rejected                        |
| Evaluate local unknown handling          | test behaviour on images outside the known training distribution      |
| Evaluate hierarchical decision-making    | test fine-label, coarse-label, and manual-review outputs              |
| Evaluate safe policy tuning              | examine the trade-off between coverage and safety                     |
| Evaluate active learning workflow        | verify whether local unknown samples can be selected for human review |
| Evaluate prototype usability             | confirm inference, backend, and frontend workflows operate correctly  |

These objectives match the main aim of OpenWaste-HR: to move beyond a normal classifier and create a safer decision workflow for real-world waste images.

## 3. Experimental Setup

The first prototype was implemented using a modular project structure. The machine learning pipeline was kept separate from the backend, frontend, documentation, and testing components.

The main experimental workflow was:

```text id="l5v4xu"
dataset preparation → baseline training → closed-set evaluation → reject-option evaluation → local unknown evaluation → hierarchical policy evaluation → safe policy tuning → active learning → inference and prototype validation
```

The evaluation used the following main artefacts:

| Artefact                     | Purpose                                                |
| ---------------------------- | ------------------------------------------------------ |
| dataset manifests            | standardise image records and labels                   |
| train/validation/test splits | separate training and evaluation data                  |
| trained checkpoint           | evaluate the trained baseline model                    |
| threshold-selection outputs  | select reject and hierarchical thresholds              |
| local unknown manifest       | evaluate behaviour on locally collected unknown images |
| metrics files                | record quantitative results                            |
| frontend/backend demo        | validate prototype usability                           |

All major components were supported by automated tests to reduce the risk of implementation errors.

## 4. Dataset and Splits

The known-class baseline was trained using a TrashNet-style dataset structure mapped into the OpenWaste-HR taxonomy.

The generated split sizes were:

| Split      | Samples |
| ---------- | ------: |
| Train      |    1766 |
| Validation |     377 |
| Test       |     384 |
| Total      |    2527 |

The trained model currently uses five known fine labels:

| Fine Label      | Coarse Label |
| --------------- | ------------ |
| paper_cardboard | recyclable   |
| plastic         | recyclable   |
| glass           | recyclable   |
| metal           | recyclable   |
| residual        | residual     |

The dataset inspection showed that the current known dataset is useful for a first baseline, but it has limited taxonomy coverage. It does not yet fully cover target categories such as organic and e_waste_battery. This limitation is important because real waste images may contain materials or objects not represented in the current training set.

A separate local unknown dataset was also used for open-set evaluation. This dataset contains 40 local images treated as unknown-test samples.

## 5. Evaluation Metrics

Different metrics were used for different parts of the evaluation.

For closed-set classification, the main metrics were:

| Metric            | Purpose                                                |
| ----------------- | ------------------------------------------------------ |
| accuracy          | overall forced-classification correctness              |
| balanced accuracy | class-balanced performance                             |
| macro-F1          | average F1 across classes without class-size weighting |
| weighted-F1       | F1 weighted by class frequency                         |

For reject-option and hierarchical evaluation, the main metrics were:

| Metric                               | Purpose                                                           |
| ------------------------------------ | ----------------------------------------------------------------- |
| coverage                             | proportion of samples accepted automatically                      |
| rejection rate                       | proportion of samples sent to manual review                       |
| selective accuracy                   | accuracy among accepted samples only                              |
| selective macro-F1                   | macro-F1 among accepted samples                                   |
| hierarchical success rate            | success after allowing fine or coarse correctness                 |
| unknown rejection/manual-review rate | proportion of local unknown samples rejected or routed to review  |
| unknown false acceptance rate        | proportion of local unknown samples incorrectly accepted as known |

These metrics are more suitable than accuracy alone because OpenWaste-HR is designed for uncertainty-aware classification.

## 6. Closed-Set Baseline Evaluation

The first baseline was trained as a closed-set image classifier. In a closed-set setting, the model must assign every image to one of the known fine labels.

The baseline produced the following test results:

| Metric            |    Value |
| ----------------- | -------: |
| Test accuracy     | 0.692708 |
| Balanced accuracy | 0.654500 |
| Macro-F1          | 0.645600 |
| Weighted-F1       | 0.700900 |

These results provide the first baseline for comparison. However, closed-set classification is limited because it always forces a known label, even if the image is uncertain, ambiguous, mixed, or outside the known label set.

Therefore, the closed-set baseline is useful as a starting point, but it is not sufficient for the project goal.

## 7. Reject-Option Baseline Evaluation

Reject-option baselines were implemented to reduce unsafe forced predictions.

Three methods were evaluated:

| Method               | Score Used                    |
| -------------------- | ----------------------------- |
| confidence threshold | maximum softmax probability   |
| max-logit score      | maximum raw logit value       |
| energy score         | energy-based confidence score |

The confidence-threshold reject baseline selected a threshold of 0.640000 using validation data.

On the known test set, the confidence reject baseline achieved:

| Metric                    |    Value |
| ------------------------- | -------: |
| Known-test coverage       | 0.682292 |
| Known-test rejection rate | 0.317708 |
| Selective accuracy        | 0.770992 |
| Selective macro-F1        | 0.715164 |

This shows that rejecting uncertain samples improves reliability among accepted predictions. The selective accuracy of 0.770992 is higher than the closed-set accuracy of 0.692708.

However, this improvement comes with reduced coverage because 31.7708% of known-test samples were rejected.

## 8. Local Unknown Evaluation

The local unknown evaluation tested whether the reject methods could detect images outside the known training distribution.

The confidence reject baseline produced:

| Metric                        |    Value |
| ----------------------------- | -------: |
| Local unknown samples         |       40 |
| Rejected unknown samples      |       14 |
| Accepted unknown samples      |       26 |
| Unknown rejection rate        | 0.350000 |
| Unknown false acceptance rate | 0.650000 |

The max-logit and energy baselines were also evaluated:

| Method               | Unknown Rejection Rate | Unknown False Acceptance Rate |
| -------------------- | ---------------------: | ----------------------------: |
| Confidence threshold |               0.350000 |                      0.650000 |
| Max-logit score      |               0.275000 |                      0.725000 |
| Energy score         |               0.200000 |                      0.800000 |

The confidence threshold performed best on the local unknown dataset among these three simple reject methods.

However, even the best simple reject method still accepted 26 out of 40 local unknown samples. This result supports the need for safer decision policies and active learning.

## 9. Hierarchical Decision Policy Evaluation

The first hierarchical decision policy was implemented to allow three possible decision types:

| Decision Type | Meaning                                 |
| ------------- | --------------------------------------- |
| fine_label    | accept a detailed known fine label      |
| coarse_label  | return a broader safe fallback category |
| manual_review | route the image to human review         |

The first hierarchical policy achieved the following known-test result:

| Metric                             |    Value |
| ---------------------------------- | -------: |
| Known total samples                |      384 |
| Fine decisions                     |      262 |
| Coarse fallback decisions          |       96 |
| Manual-review decisions            |       26 |
| Known decision coverage            | 0.932292 |
| Hierarchical success over accepted | 0.824022 |

This was a useful result because the system could make more decisions than the confidence reject baseline by using coarse fallback labels.

However, the local unknown result showed that the policy was too permissive:

| Metric                  |    Value |
| ----------------------- | -------: |
| Local unknown samples   |       40 |
| Fine accepts            |       26 |
| Coarse accepts          |       11 |
| Manual-review count     |        3 |
| Manual-review rate      | 0.075000 |
| Unknown acceptance rate | 0.925000 |

This means that the first hierarchical policy had high known-test coverage but weak unknown safety. Only 3 out of 40 local unknown images were routed to manual review.

## 10. Safe Hierarchical Policy Tuning

A safe hierarchical tuning stage was implemented to select stricter thresholds.

The selected safe policy used:

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.900000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.650000 |
| minimum_confidence_for_coarse | 0.650000 |

The known-test result was:

| Metric                             |    Value |
| ---------------------------------- | -------: |
| Known total samples                |      384 |
| Fine decisions                     |      145 |
| Coarse fallback decisions          |      108 |
| Manual-review decisions            |      131 |
| Known decision coverage            | 0.658854 |
| Hierarchical success over accepted | 0.889328 |

The local unknown result was:

| Metric                  |    Value |
| ----------------------- | -------: |
| Local unknown samples   |       40 |
| Fine accepts            |       16 |
| Coarse accepts          |        9 |
| Manual-review count     |       15 |
| Manual-review rate      | 0.375000 |
| Unknown acceptance rate | 0.625000 |

The safe policy improved the local unknown manual-review rate from 0.075000 to 0.375000 when compared with the first hierarchical policy.

This shows a clear safety-coverage trade-off. The safe policy accepts fewer known-test samples, but the accepted decisions are more reliable and more local unknown samples are routed to review.

## 11. Policy Comparison

The main policy comparison is shown below.

| System                     | Known Coverage | Accepted Reliability | Local Unknown Review/Rejection Rate | Local Unknown Acceptance Rate |
| -------------------------- | -------------: | -------------------: | ----------------------------------: | ----------------------------: |
| Closed-set baseline        |       1.000000 |             0.692708 |                            0.000000 |                      1.000000 |
| Confidence reject baseline |       0.682292 |             0.770992 |                            0.350000 |                      0.650000 |
| Hierarchical policy v1     |       0.932292 |             0.824022 |                            0.075000 |                      0.925000 |
| Safe hierarchical policy   |       0.658854 |             0.889328 |                            0.375000 |                      0.625000 |

The closed-set baseline gives full coverage but no unknown safety.

The confidence reject baseline improves unknown rejection but only returns fine labels or rejection.

The first hierarchical policy gives strong known-test coverage and supports coarse fallback, but accepts too many local unknown images.

The safe hierarchical policy is selected as the current OpenWaste-HR policy because it provides the best safety-oriented balance for the current prototype.

## 12. Active Learning Evaluation

The active learning stage was evaluated as a workflow rather than as a retrained-model result.

The system selected 20 local unknown candidates for human review.

| Candidate Type            | Count |
| ------------------------- | ----: |
| Manual-review candidates  |    12 |
| Coarse-label candidates   |     4 |
| Fine-label candidates     |     4 |
| Total selected candidates |    20 |

A human labelling sheet was generated for these 20 candidates. Since the sheet has not yet been filled, the reviewed-label processing stage currently reports:

| Metric                 | Value |
| ---------------------- | ----: |
| Total rows             |    20 |
| Reviewed rows          |     0 |
| Pending review rows    |    20 |
| Ready for dataset rows |     0 |

This confirms that the active learning loop is structurally implemented, but the next stage requires manual annotation before retraining or dataset-v2 comparison.

## 13. Inference and Prototype Validation

The system was also evaluated at prototype level.

The demo image was:

```text id="w1hr7z"
ml/data/local_unknown/images/local_000001.jpg
```

The inference result was:

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

The same result was successfully returned through:

| Interface                           | Status  |
| ----------------------------------- | ------- |
| single-image command-line inference | working |
| batch inference                     | working |
| prototype API wrapper               | working |
| FastAPI backend endpoint            | working |
| simple frontend demo                | working |

This confirms that OpenWaste-HR is not only an offline experiment. It is now a working end-to-end prototype.

## 14. Automated Testing Result

Automated testing was used throughout development.

The latest full test suite before this evaluation draft passed:

| Metric            | Value |
| ----------------- | ----: |
| Tests passed      |   132 |
| Blocking failures |     0 |

After this evaluation chapter draft is added, the expected test count is 138.

The tests cover dataset processing, taxonomy validation, model utilities, evaluation metrics, hierarchical decision logic, active learning, human labelling workflow, inference, API wrapper, backend endpoint, frontend files, documentation, and architecture files.

## 15. Discussion

The evaluation shows that OpenWaste-HR provides value beyond a standard closed-set classifier.

The closed-set baseline achieved 0.692708 test accuracy, but it cannot identify when an image should not be automatically accepted. The confidence reject baseline improved selective accuracy and rejected 35% of local unknown samples, but still accepted many unknown images.

The hierarchical policy introduced a more flexible decision structure by allowing fine-label, coarse-label, and manual-review outputs. The first policy had high known-test coverage, but weak unknown handling. The safe tuned policy improved local unknown manual-review behaviour and achieved higher reliability among accepted decisions.

This supports the main argument of the project: waste classification systems should not rely only on forced fine-label predictions. In practical settings, uncertain images should be handled through safer fallback and review mechanisms.

## 16. Limitations of the Current Evaluation

The current evaluation has several limitations:

| Limitation                                      | Explanation                                              |
| ----------------------------------------------- | -------------------------------------------------------- |
| Dataset coverage is limited                     | only five known fine labels are currently trained        |
| Additional categories are not fully represented | organic and e_waste_battery need more training data      |
| Local unknown dataset is small                  | current local unknown evaluation uses 40 samples         |
| Human labels are not filled yet                 | active learning candidates are pending annotation        |
| No retrained active-learning model yet          | reviewed labels must be added before retraining          |
| First baseline model only                       | pretrained and expanded-dataset models are still planned |
| Frontend uses image paths                       | direct image upload is not yet implemented               |

These limitations will be addressed in later evaluation stages.

## 17. Planned Future Evaluation

The next evaluation stages should include:

| Future Evaluation                         | Purpose                                                        |
| ----------------------------------------- | -------------------------------------------------------------- |
| Pretrained model training                 | compare transfer learning against the current scratch baseline |
| Additional public datasets                | improve taxonomy coverage and dataset diversity                |
| Human-labelled local samples              | create a reviewed local dataset v2                             |
| Retraining with local feedback            | compare active-learning v2 model against baseline              |
| Calibration and improved open-set scoring | reduce unknown false acceptance                                |
| Final model comparison table              | compare all models and decision policies                       |

The intended future comparison is:

| Model/System              | Purpose                                                    |
| ------------------------- | ---------------------------------------------------------- |
| Baseline A                | current scratch-trained TrashNet-style model               |
| Baseline B                | pretrained transfer-learning model                         |
| Dataset-expanded model    | model trained with additional public datasets              |
| Active-learning v2 model  | model updated with reviewed local samples                  |
| Final OpenWaste-HR policy | best model combined with safe hierarchical decision policy |

## 18. Conclusion

The current evaluation demonstrates that OpenWaste-HR is a working hierarchical open-set waste classification prototype.

The results show that closed-set accuracy alone is not enough for real-world waste classification. The safe hierarchical policy provides a more appropriate workflow by supporting fine-label prediction, coarse fallback, and manual-review routing.

Although future improvements are still needed, the current prototype successfully demonstrates the core research direction: safer waste classification through hierarchical open-set decision-making and local active learning support.
