# Human Labelling Sheet v1

## Purpose

This stage creates a human labelling sheet for the active learning candidates.

The active learning stage selected local unknown dataset images that are useful for human review. This sheet converts those selected candidates into a structured annotation file.

## Why This Is Needed

OpenWaste-HR is designed for open-world waste classification. The system should not only classify images; it should also identify uncertain or risky cases that need human feedback.

The human labelling sheet supports this feedback loop.

## Input

The input is:

* active_learning_candidates_v1.csv

This file contains the selected local unknown candidate images ranked by active learning score.

## Output

The output is:

* human_labelling_sheet_v1.csv

This file includes model information and empty human annotation fields.

## Human Annotation Fields

| Column             | Meaning                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| human_decision     | human decision such as known_label, new_unknown_class, mixed_waste, unclear_image, remove_sample |
| human_fine_label   | selected fine label if the item fits an existing known class                                     |
| human_coarse_label | selected coarse label if a broader category is suitable                                          |
| proposed_new_label | suggested new label if the item does not fit the existing taxonomy                               |
| human_confidence   | low, medium, or high                                                                             |
| human_notes        | explanation from the reviewer                                                                    |
| reviewed_by        | reviewer name or initials                                                                        |
| review_date        | date of review                                                                                   |

## Allowed Fine Labels

| Fine Label      |
| --------------- |
| paper_cardboard |
| plastic         |
| glass           |
| metal           |
| organic         |
| e_waste_battery |
| residual        |
| unknown         |

## Allowed Coarse Labels

| Coarse Label  |
| ------------- |
| recyclable    |
| organic       |
| hazardous     |
| residual      |
| unknown       |
| manual_review |

## Research Meaning

This stage turns active learning from a concept into a usable project workflow.

The model identifies useful samples, and the human labelling sheet prepares those samples for review. Later, reviewed samples can be used to improve taxonomy coverage, local data quality, threshold tuning, and open-set/manual-review behaviour.
