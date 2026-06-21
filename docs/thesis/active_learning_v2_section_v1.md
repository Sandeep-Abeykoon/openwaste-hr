# Active Learning v2 Thesis Section Draft

## Active Learning and Human-in-the-Loop Dataset Improvement

A key part of OpenWaste-HR is that the system is designed for real-world waste classification, where new and unfamiliar objects may appear after deployment. Public waste datasets are useful for initial model training, but they do not fully represent local waste conditions, local object types, damaged items, mixed materials, or unfamiliar disposal objects. Therefore, a static closed-set classifier is not enough for the intended project context.

To address this limitation, OpenWaste-HR includes an active learning workflow. The purpose of this workflow is not only to improve classification accuracy, but also to identify uncertain, unsafe, or locally unfamiliar samples that require human review. These reviewed samples can then be used to improve the dataset, update the taxonomy, or strengthen future open-set evaluation.

## Local Unknown Review Workflow

The active learning workflow begins by selecting candidate samples from the local unknown image set. These samples are prioritised using model uncertainty, hierarchical decision behaviour, and active learning scores. The selected images are written into a human labelling sheet so that a reviewer can inspect them manually.

The human review process records fields such as:

| Field              | Purpose                                                             |
| ------------------ | ------------------------------------------------------------------- |
| sample_id          | identifies the local image                                          |
| image_path         | stores the local file path                                          |
| human_decision     | records whether the image is inside or outside the current taxonomy |
| human_fine_label   | stores the corrected known fine label when applicable               |
| human_coarse_label | stores the corrected broad category when applicable                 |
| proposed_new_label | stores a possible new class name                                    |
| human_confidence   | records reviewer confidence                                         |
| human_notes        | records the reason for the decision                                 |
| reviewed_by        | records the reviewer                                                |
| review_date        | records the date of review                                          |

This structure allows the system to separate known-class corrections from true local unknowns. This is important because forcing every reviewed object into an existing known class would create label noise.

## Reviewed Local Seed Example

The first reviewed local seed in this project is:

```text id="5h52k2"
local_000001
```

This image was visually identified as a rubber slipper / flip-flop. It is not part of the current known fine-label taxonomy, which contains paper_cardboard, plastic, glass, metal, organic, e_waste_battery, and residual.

The best pretrained safe hierarchical policy predicted this sample as a known recyclable object:

| Field                  | Value           |
| ---------------------- | --------------- |
| predicted fine label   | paper_cardboard |
| max softmax confidence | 0.993654        |
| final decision type    | coarse_label    |
| final label            | recyclable      |
| final confidence       | 0.999999        |

However, the human observation shows that the object is outside the current known taxonomy. Therefore, the reviewed label seed records:

| Field                 | Value                          |
| --------------------- | ------------------------------ |
| sample_id             | local_000001                   |
| human_observed_object | rubber slipper / flip-flop     |
| human_taxonomy_status | outside_current_known_taxonomy |
| recommended_action    | keep_as_unknown_test           |
| proposed_new_label    | rubber_slipper_flip_flop       |

## Active Learning v2 Dataset Decision

The reviewed sample is not added to the known training set. Instead, the active learning v2 dataset plan assigns it the following role:

| Dataset Role                      | Value                                   |
| --------------------------------- | --------------------------------------- |
| include_in_known_training_v2      | false                                   |
| include_in_unknown_test_v2        | true                                    |
| include_as_future_class_candidate | true                                    |
| active_learning_v2_role           | unknown_test_and_future_class_candidate |

This decision prevents the rubber slipper sample from being incorrectly used as paper_cardboard, plastic, glass, metal, organic, e_waste_battery, or residual. It remains useful as an open-set evaluation sample and as a possible future class candidate if more similar local examples are collected.

## Research Importance

This result supports the central motivation of OpenWaste-HR. The model can produce high confidence even for a local object outside the training taxonomy. This shows that closed-set confidence alone is not sufficient for safe real-world waste classification.

The active learning workflow improves the system by allowing uncertain or unfamiliar local samples to be reviewed and assigned appropriate dataset roles. Some reviewed samples may become known-class training examples, while others should remain unknown-test examples or future class candidates.

This supports the project contribution in four ways:

1. It connects local unknown evaluation to dataset improvement.
2. It avoids forcing unfamiliar objects into incorrect known labels.
3. It supports future taxonomy expansion.
4. It creates a human-in-the-loop process for safer deployment.

## Summary

The active learning v2 stage shows that OpenWaste-HR is not only a classifier. It is a complete workflow for hierarchical open-set waste classification, local unknown detection, manual review, and future dataset improvement.

The current reviewed seed is small, but it demonstrates the full process. As more local images are reviewed, the same workflow can be used to build a stronger local dataset, improve open-set evaluation, and support future retraining.
