# Final Project Summary After Expanded Public Evaluation v1

## Project Summary

OpenWaste-HR is a hierarchical open-set waste classification system with a reject option and local active learning workflow. The project was designed to address a key limitation in normal waste classification models: they usually behave as closed-set classifiers and force every input into a known waste category, even when the item may be outside the model taxonomy.

The final system therefore supports three types of decisions:

| Decision        | Meaning                                   |
| --------------- | ----------------------------------------- |
| fine label      | confident fine-grained class prediction   |
| coarse fallback | safer higher-level waste category         |
| manual review   | uncertain, unsafe, or unknown-like sample |

## Dataset Progression

The project began with a TrashNet-only workflow and was later expanded with RealWaste.

| Stage                 | Dataset Scope                                          |
| --------------------- | ------------------------------------------------------ |
| Initial baseline      | TrashNet-style known waste classes                     |
| Expanded public model | TrashNet-style known samples + RealWaste known samples |
| Unknown evaluation    | local unknown dataset + RealWaste Textile Trash        |

The expanded public dataset introduced a broader six-class known taxonomy:

* paper/cardboard
* plastic
* glass
* metal
* organic
* residual

RealWaste `Textile Trash` was not added as a known class. It was intentionally kept as public unknown/future-class evaluation data.

## Model Comparison

| Model                            | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| -------------------------------- | -------: | ----------------: | -------: | ----------: |
| Scratch TrashNet-only baseline   |   0.6927 |            0.6545 |   0.6456 |      0.7009 |
| Pretrained TrashNet-only model   |   0.8880 |            0.8431 |   0.8510 |      0.8873 |
| Pretrained expanded public model |   0.8876 |            0.8750 |   0.8819 |      0.8870 |

The expanded public pretrained model achieved almost the same overall accuracy as the TrashNet-only pretrained model, but improved macro-F1 from 0.8510 to 0.8819. This is important because macro-F1 better reflects balanced class performance.

## Reject-Option Finding

The expanded public reject-option evaluation showed that confidence-threshold rejection was strongest for known selective prediction.

| Method               | Coverage | Selective Macro-F1 | Selective Weighted-F1 |
| -------------------- | -------: | -----------------: | --------------------: |
| Confidence threshold |   0.7229 |             0.9732 |                0.9788 |
| Max-logit            |   0.7362 |             0.9627 |                0.9676 |
| Energy score         |   0.7181 |             0.9612 |                0.9668 |

This means confidence thresholding produced the most reliable accepted predictions on known test data.

## Unknown-Rejection Finding

Unknown evaluation showed a different pattern. Energy-score rejection was strongest for rejecting unknown samples.

| Unknown Source                    | Best Method  | Unknown Rejection Rate | False Acceptance Rate |
| --------------------------------- | ------------ | ---------------------: | --------------------: |
| Local unknown dataset             | Energy score |                 0.6750 |                0.3250 |
| Public unknown/future-class split | Energy score |                 0.6509 |                0.3491 |

This result is important because it shows that the best method for known selective prediction is not necessarily the best method for unknown rejection.

## Final Safe Hierarchical Policy

The final expanded public safe hierarchical policy selected the following thresholds:

| Parameter                     | Value |
| ----------------------------- | ----: |
| fine confidence threshold     | 0.995 |
| coarse confidence threshold   | 0.990 |
| coarse margin threshold       | 0.150 |
| minimum confidence for coarse | 0.350 |

The final policy achieved:

| Metric                             |  Value |
| ---------------------------------- | -----: |
| known coverage                     | 0.8819 |
| accepted hierarchical success rate | 0.9838 |
| local unknown manual review rate   | 0.4750 |
| local unknown acceptance rate      | 0.5250 |

## Final Interpretation

The expanded public safe hierarchical policy is the strongest balanced OpenWaste-HR system because it combines broader public training, improved macro-F1, high accepted-decision reliability, and hierarchical fine/coarse/manual-review outputs.

However, the results also show a limitation. The earlier TrashNet-only safe policy was stricter on local unknown samples, and the standalone energy-score method rejected more unknowns. Therefore, the final thesis should present the result as a trade-off rather than claiming that one system is best for every metric.

## Final Thesis Claim

OpenWaste-HR demonstrates that waste classification should not rely only on closed-set accuracy. A more practical waste classification system should use uncertainty-aware decision-making, support coarse fallback, and route uncertain or unknown-like cases to manual review. The expanded public safe hierarchical policy provides the strongest balanced result, while energy-score rejection provides a clear future direction for improving unknown safety.
