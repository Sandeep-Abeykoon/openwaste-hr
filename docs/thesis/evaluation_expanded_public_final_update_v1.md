# Evaluation Chapter Update: Expanded Public Final Comparison v1

## Expanded Public Evaluation Summary

After the original TrashNet-only experiments, the project was extended by adding RealWaste as a second public dataset. This created an expanded public dataset with six known fine labels: paper/cardboard, plastic, glass, metal, organic, and residual. The RealWaste Textile Trash category was intentionally excluded from known training and used as public unknown/future-class evaluation data.

The expanded public pretrained model achieved 0.8876 accuracy, 0.8750 balanced accuracy, 0.8819 macro-F1, and 0.8870 weighted-F1 on the expanded public known test set. Compared with the earlier TrashNet-only pretrained model, the overall accuracy remained similar, but macro-F1 improved from 0.8510 to 0.8819. This is significant because macro-F1 better reflects balanced performance across classes, especially after adding the organic class.

The reject-option evaluation showed that confidence-threshold rejection performed best for known selective prediction. It achieved 0.7229 coverage, 0.9789 selective accuracy, and 0.9732 selective macro-F1. This demonstrates that the model can produce much more reliable accepted predictions when uncertain samples are rejected.

Unknown evaluation showed a different pattern. Energy-score rejection performed best for unknown rejection. On the local unknown dataset, it rejected 27 out of 40 samples, giving a 0.6750 unknown rejection rate. On the public unknown/future-class split, it rejected 207 out of 318 Textile Trash samples, giving a 0.6509 unknown rejection rate. This shows that the strongest method for known selective prediction was not the same as the strongest method for unknown rejection.

The final expanded public safe hierarchical policy selected a fine confidence threshold of 0.995, a coarse confidence threshold of 0.990, a coarse margin threshold of 0.150, and a minimum confidence for coarse fallback of 0.350. This policy achieved 0.8819 known coverage and 0.9838 hierarchical success over accepted decisions. On the local unknown dataset, it sent 19 out of 40 samples to manual review.

These results support the main research argument of OpenWaste-HR. A waste classification system should not be treated as a simple closed-set classifier. Instead, it should support fine labels, coarse fallback, and manual review decisions. The expanded public model improved class-balanced known performance and accepted-decision reliability, while the unknown evaluation demonstrated the continuing need for explicit uncertainty-aware rejection.
