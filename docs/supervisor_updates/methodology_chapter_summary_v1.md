# Methodology Chapter Summary v1

## Purpose

This document summarises the consolidated methodology chapter for OpenWaste-HR.

The methodology explains how the project was designed, implemented, and evaluated as a hierarchical open-set waste classification prototype.

## Main Methodology Flow

```text id="68dsnv"
taxonomy design → dataset preparation → baseline training → reject-option evaluation → local unknown evaluation → hierarchical decision policy → safe policy tuning → active learning → inference → backend/frontend prototype
```

## Methodology Sections

| Section                    | Purpose                                                        |
| -------------------------- | -------------------------------------------------------------- |
| Research design            | explains the experimental prototype approach                   |
| Taxonomy methodology       | defines fine labels, coarse labels, unknown, and manual_review |
| Dataset methodology        | explains manifest-based dataset preparation                    |
| Dataset inspection         | checks images, labels, imbalance, and missing files            |
| Baseline training          | trains the first closed-set classifier                         |
| Closed-set evaluation      | measures forced-classification performance                     |
| Reject-option methodology  | adds confidence, max-logit, and energy rejection               |
| Local unknown evaluation   | tests behaviour on local unknown images                        |
| Hierarchical decision      | outputs fine_label, coarse_label, or manual_review             |
| Safe policy tuning         | selects stricter thresholds for safer decisions                |
| Active learning            | selects local candidates for human review                      |
| Human labelling            | prepares reviewed samples for future dataset updates           |
| Inference                  | supports single-image, batch, and API-style inference          |
| Backend/frontend prototype | demonstrates the system end-to-end                             |
| Testing                    | supports reproducibility and implementation confidence         |

## Key Methodological Contribution

The main contribution is not only classification.

The methodology combines:

* hierarchical taxonomy
* reject/manual-review decisions
* local unknown evaluation
* safety-focused threshold tuning
* local active learning
* human labelling workflow
* backend/frontend prototype validation

## Why This Methodology Fits the Problem

Waste classification in real environments is open-world. Images may contain unclear, mixed, damaged, unfamiliar, or locally specific items.

Therefore, the system should not always force a known label. The methodology supports safer outputs through:

| Output        | Meaning                               |
| ------------- | ------------------------------------- |
| fine_label    | confident detailed known label        |
| coarse_label  | safer broad fallback                  |
| manual_review | uncertain sample sent to human review |

## Future Methodology Work

The next methodology extensions are:

1. train a pretrained transfer-learning model
2. add additional public waste datasets
3. fill human correction labels
4. create reviewed local dataset v2
5. retrain or fine-tune using active-learning feedback
6. compare improved models with the original baseline
7. update the thesis evaluation with final comparison tables
