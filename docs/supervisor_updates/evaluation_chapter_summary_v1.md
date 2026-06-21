# Evaluation Chapter Summary v1

## Current Evaluation Status

The current OpenWaste-HR evaluation chapter draft has been created for the first working prototype.

This evaluation is not the final comparison stage. It summarises the current baseline, reject-option results, hierarchical policy results, local unknown evaluation, active learning workflow, and prototype validation.

## Main Results

| Area                                                | Result   |
| --------------------------------------------------- | -------- |
| Closed-set baseline accuracy                        | 0.692708 |
| Confidence reject selective accuracy                | 0.770992 |
| Confidence reject local unknown rejection rate      | 0.350000 |
| First hierarchical policy known coverage            | 0.932292 |
| First hierarchical local unknown manual-review rate | 0.075000 |
| Safe hierarchical known coverage                    | 0.658854 |
| Safe hierarchical accepted reliability              | 0.889328 |
| Safe hierarchical local unknown manual-review rate  | 0.375000 |
| Active learning candidates selected                 | 20       |
| Prototype demo                                      | working  |

## Key Interpretation

The evaluation shows that OpenWaste-HR should not be judged only as a normal classifier.

The current baseline achieves moderate closed-set accuracy, but closed-set classification always forces a label. The reject-option and hierarchical policy results show why a safer decision workflow is useful.

The safe hierarchical policy is currently selected because it improves accepted-decision reliability and routes more local unknown images to manual review.

## Current Selected Policy

| Threshold                     |    Value |
| ----------------------------- | -------: |
| fine_confidence_threshold     | 0.900000 |
| coarse_confidence_threshold   | 0.800000 |
| coarse_margin_threshold       | 0.650000 |
| minimum_confidence_for_coarse | 0.650000 |

## Current Prototype Interfaces

| Interface                | Status  |
| ------------------------ | ------- |
| single-image inference   | working |
| batch inference          | working |
| prototype API wrapper    | working |
| FastAPI backend endpoint | working |
| frontend demo            | working |

## Limitations

| Limitation                               | Explanation                                               |
| ---------------------------------------- | --------------------------------------------------------- |
| trained labels are limited               | only five known fine labels are currently trained         |
| local unknown dataset is small           | current local unknown evaluation uses 40 samples          |
| human review is pending                  | active learning candidates are selected but not annotated |
| pretrained comparison is pending         | current model is still the first baseline                 |
| additional dataset comparison is pending | more public datasets will be added later                  |

## Next Evaluation Work

The next evaluation stages should include:

1. train a pretrained transfer-learning model
2. add additional public waste datasets
3. fill the human labelling sheet
4. create a reviewed local dataset v2
5. retrain or fine-tune using reviewed local labels
6. compare all models against the original baseline
7. update the evaluation chapter with final comparison tables

## Thesis Message

The current results support the project argument:

```text id="gbluqy"
OpenWaste-HR is not only a waste classifier; it is a hierarchical open-set decision workflow that can return fine labels, coarse fallback labels, or manual-review decisions under uncertainty.
```
