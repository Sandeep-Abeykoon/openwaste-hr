# OpenWaste-HR Final Prototype Summary v1

## Purpose

This document summarises the current OpenWaste-HR prototype.

OpenWaste-HR is a hierarchical open-set waste classification prototype. Its purpose is not only to classify waste images, but also to make safer decisions when the model is uncertain or when an image may not belong to the known training classes.

## Current Prototype Status

The current prototype supports the full workflow:

```text
dataset preparation → baseline training → reject option → hierarchical decision policy → safe policy tuning → local unknown evaluation → active learning → inference → backend → frontend demo
```

## Implemented Components

| Component                           | Status   | Purpose                                                                     |
| ----------------------------------- | -------- | --------------------------------------------------------------------------- |
| Project structure                   | complete | organised ML, backend, frontend, docs, tests, and outputs                   |
| Taxonomy                            | complete | defines fine labels, coarse labels, unknown, and manual-review concepts     |
| Dataset manifest system             | complete | standardises dataset records and splits                                     |
| TrashNet manifest builder           | complete | creates train, validation, and test splits from TrashNet-style folders      |
| Dataset inspection                  | complete | checks labels, image sizes, missing files, and class distribution           |
| Baseline classifier                 | complete | trains a MobileNetV3-style image classifier                                 |
| Closed-set evaluation               | complete | evaluates normal forced classification performance                          |
| Confidence reject baseline          | complete | adds simple manual-review rejection using softmax confidence                |
| Max-logit and energy baselines      | complete | compares alternative open-set scoring methods                               |
| Local unknown dataset evaluation    | complete | tests how reject methods behave on local unknown images                     |
| Hierarchical policy v1              | complete | returns fine label, coarse fallback, or manual review                       |
| Safe hierarchical tuning            | complete | tunes stricter thresholds for safer unknown handling                        |
| Policy comparison                   | complete | compares closed-set, reject baseline, hierarchical policy, and tuned policy |
| Active learning candidate selection | complete | selects useful local unknown images for human review                        |
| Human labelling sheet               | complete | creates a structured annotation sheet for selected candidates               |
| Reviewed label processing           | complete | processes filled human labels into dataset actions                          |
| Single-image inference              | complete | runs OpenWaste-HR on one image                                              |
| Batch inference                     | complete | runs OpenWaste-HR on a folder of images                                     |
| Prototype API wrapper               | complete | returns backend-friendly request/response JSON                              |
| FastAPI backend endpoint            | complete | exposes `/health` and `/api/inference/predict`                              |
| Simple frontend demo                | complete | allows browser-based prototype testing                                      |
| Prototype run guide                 | complete | explains how to run the demo                                                |
| Supervisor checklist                | complete | gives a structured demo script                                              |

## Dataset Summary

The current known-class baseline uses a TrashNet-style dataset structure.

| Dataset Split | Samples |
| ------------- | ------: |
| Train         |    1766 |
| Validation    |     377 |
| Test          |     384 |
| Total         |    2527 |

Known fine labels used by the trained model:

| Fine Label      |
| --------------- |
| paper_cardboard |
| plastic         |
| glass           |
| metal           |
| residual        |

The initial known dataset is limited because it does not fully cover all target taxonomy categories such as organic and e_waste_battery.

## Baseline Classifier Result

The baseline classifier was trained and evaluated as a closed-set model.

| Metric            |    Value |
| ----------------- | -------: |
| Test accuracy     | 0.692708 |
| Balanced accuracy | 0.654500 |
| Macro-F1          | 0.645600 |
| Weighted-F1       | 0.700900 |

This result is useful as a starting baseline, but closed-set classification is not enough for open-world waste classification because it always forces a known label.

## Confidence Reject Baseline Result

The confidence-threshold reject baseline selected a threshold of 0.640000.

| Metric                    |    Value |
| ------------------------- | -------: |
| Known-test coverage       | 0.682292 |
| Known-test rejection rate | 0.317708 |
| Selective accuracy        | 0.770992 |
| Selective macro-F1        | 0.715164 |

Local unknown dataset result:

| Metric                        |    Value |
| ----------------------------- | -------: |
| Local unknown samples         |       40 |
| Rejected unknown samples      |       14 |
| Accepted unknown samples      |       26 |
| Unknown rejection rate        | 0.350000 |
| Unknown false acceptance rate | 0.650000 |

This shows that simple rejection improves safety, but still accepts many unknown local images as known classes.

## Hierarchical Decision Policy Result

The first hierarchical policy returned fine labels, coarse fallback labels, or manual review.

Known-test result:

| Metric                             |    Value |
| ---------------------------------- | -------: |
| Known total samples                |      384 |
| Fine decisions                     |      262 |
| Coarse fallback decisions          |       96 |
| Manual-review decisions            |       26 |
| Known decision coverage            | 0.932292 |
| Hierarchical success over accepted | 0.824022 |

Local unknown result:

| Metric                  |    Value |
| ----------------------- | -------: |
| Local unknown samples   |       40 |
| Fine accepts            |       26 |
| Coarse accepts          |       11 |
| Manual-review count     |        3 |
| Manual-review rate      | 0.075000 |
| Unknown acceptance rate | 0.925000 |

This policy gave high known-test coverage, but it was too permissive for local unknown images.

## Safe Hierarchical Policy Tuning Result

The tuned safe hierarchical policy selected stricter thresholds.

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.900000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.650000 |
| minimum_confidence_for_coarse | 0.650000 |

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

The safe tuned policy is selected as the current OpenWaste-HR decision policy because it better supports uncertainty-aware decisions.

## Active Learning Workflow Result

The active learning stage selected 20 local unknown candidates for human labelling.

| Candidate Type            | Count |
| ------------------------- | ----: |
| Manual-review candidates  |    12 |
| Coarse-label candidates   |     4 |
| Fine-label candidates     |     4 |
| Total selected candidates |    20 |

The human labelling sheet was created with 20 rows. Since it has not yet been manually filled, reviewed label processing currently reports:

| Metric                 | Value |
| ---------------------- | ----: |
| Total rows             |    20 |
| Reviewed rows          |     0 |
| Pending review rows    |    20 |
| Ready for dataset rows |     0 |

## Inference and Prototype Result

The current demo image is:

```text
ml/data/local_unknown/images/local_000001.jpg
```

Single-image inference result:

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
| command-line single-image inference | working |
| batch inference                     | working |
| prototype API wrapper               | working |
| FastAPI backend endpoint            | working |
| simple frontend demo page           | working |

## Backend and Frontend Demo

The backend exposes:

| Method | Endpoint               | Purpose                     |
| ------ | ---------------------- | --------------------------- |
| GET    | /health                | checks backend status       |
| POST   | /api/inference/predict | runs OpenWaste-HR inference |

The frontend allows the user to enter:

| Field       | Example                                       |
| ----------- | --------------------------------------------- |
| Image path  | ml/data/local_unknown/images/local_000001.jpg |
| Sample ID   | local_000001                                  |
| Backend URL | http://127.0.0.1:8000                         |

The browser displays the final decision, model prediction, confidence values, class probabilities, and raw JSON response.

## Main Research Contribution Demonstrated

The prototype demonstrates that OpenWaste-HR is not only a normal waste classifier.

It implements a safer decision workflow:

```text
fine-label prediction → coarse fallback if needed → manual_review if unsafe or uncertain
```

The main contribution is the combination of:

* hierarchical fine/coarse taxonomy
* reject/manual-review decision option
* local unknown evaluation
* safe threshold tuning
* active learning candidate selection
* human labelling workflow
* usable inference, backend, and frontend prototype

## Current Limitations

| Limitation                   | Explanation                                                |
| ---------------------------- | ---------------------------------------------------------- |
| Dataset coverage             | current trained model uses only five known fine labels     |
| Local unknown handling       | tuned policy still accepts some unknown images             |
| Human labels not filled yet  | active learning sheet exists, but annotations are pending  |
| Image upload not implemented | frontend currently uses image paths instead of file upload |
| Baseline model only          | stronger pretrained models may improve performance         |

## Next Steps

Recommended next improvements:

1. add frontend image upload
2. process filled human labels after review
3. create dataset v2 from reviewed local labels
4. train a stronger pretrained model
5. compare improved model against the current baseline
6. improve open-set scoring and calibration
7. prepare thesis implementation and evaluation chapter using these results

## Conclusion

OpenWaste-HR is now a working end-to-end prototype.

The project includes experimental evaluation, safer hierarchical decision-making, active learning support, and a usable backend/frontend demonstration. This makes it suitable for supervisor demonstration and further thesis development.
