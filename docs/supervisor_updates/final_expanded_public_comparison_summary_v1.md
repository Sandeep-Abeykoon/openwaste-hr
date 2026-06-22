# Final Expanded Public Comparison Summary v1

## Purpose

This summary explains the final comparison after adding RealWaste and training the expanded public pretrained model.

## Main Finding

The expanded public pretrained model is stronger for class-balanced known classification and accepted-decision reliability.

## Closed-Set Comparison

| Model                            | Accuracy | Macro-F1 |
| -------------------------------- | -------: | -------: |
| TrashNet-only pretrained model   |   0.8880 |   0.8510 |
| Expanded public pretrained model |   0.8876 |   0.8819 |

## Reject-Option Finding

| Method               | Best Use                   |
| -------------------- | -------------------------- |
| Confidence threshold | known selective prediction |
| Energy score         | unknown rejection          |

## Final Safe Policy Comparison

| Policy                      | Known Coverage | Accepted Success Rate | Local Unknown Manual Review Rate |
| --------------------------- | -------------: | --------------------: | -------------------------------: |
| TrashNet-only safe policy   |         0.8646 |                0.9608 |                           0.6000 |
| Expanded public safe policy |         0.8819 |                0.9838 |                           0.4750 |

## Supervisor Explanation

The expanded public safe policy gives better known-class usability and higher accepted-decision reliability. However, the earlier safe policy was stricter on local unknown samples.

Therefore, the final thesis should present the expanded public safe hierarchical policy as the strongest balanced system, while also explaining that energy-score rejection is the strongest standalone unknown-rejection method.
