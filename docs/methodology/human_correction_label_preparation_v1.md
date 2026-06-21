# Human Correction Label Preparation v1

## Purpose

This stage prepares the OpenWaste-HR project for human correction labels and active learning v2.

The earlier active-learning stage selected uncertain or unsafe local unknown samples for review. This stage documents how those samples should be reviewed before they are used for future dataset improvement.

## Why Human Correction Is Needed

The current best model is the pretrained safe hierarchical policy. It improves known-test reliability and local unknown handling, but it can still assign a known class or coarse fallback to a real unknown object.

For example, the image:

```text
ml/data/local_unknown/images/local_000001.jpg
```

was visually identified as a rubber slipper / flip-flop. This object is not part of the current known fine-label taxonomy.

However, the best active prototype predicted:

| Field                  | Value           |
| ---------------------- | --------------- |
| Predicted fine label   | paper_cardboard |
| Max softmax confidence | 0.993654        |
| Final decision type    | coarse_label    |
| Final label            | recyclable      |
| Final confidence       | 0.999999        |

This shows why human correction labels are needed. The model can be highly confident even when the object is outside the known training categories.

## Current Known Fine Labels

The current known fine-label taxonomy contains:

| Fine Label      | Coarse Label |
| --------------- | ------------ |
| paper_cardboard | recyclable   |
| plastic         | recyclable   |
| glass           | recyclable   |
| metal           | recyclable   |
| organic         | organic      |
| e_waste_battery | hazardous    |
| residual        | residual     |

## Local Unknown Review Rule

When a reviewer sees an object that is not clearly one of the known fine labels, the sample should not be forced into a known class.

Instead, it should be marked as:

```text
outside_current_known_taxonomy
```

and routed as a local unknown / manual-review candidate.

## Review Actions

| Human Observation                  | Recommended Action                     |
| ---------------------------------- | -------------------------------------- |
| clearly matches a known fine label | assign that fine label                 |
| only broad category is clear       | assign coarse label and mark uncertain |
| outside current taxonomy           | keep as local unknown                  |
| poor image quality                 | mark needs recollection                |
| duplicate image                    | mark duplicate                         |
| unsafe or ambiguous item           | mark manual review                     |

## Active Learning v2 Meaning

The reviewed labels will later support active learning v2.

Possible active-learning outcomes include:

| Outcome                       | Meaning                                            |
| ----------------------------- | -------------------------------------------------- |
| add_to_existing_class         | use the image to strengthen an existing fine class |
| keep_as_unknown_test          | keep the image for open-set testing                |
| create_future_class_candidate | save the object as a possible new class            |
| recollect_image               | capture a clearer replacement image                |
| exclude_duplicate             | remove duplicate or unusable image                 |

## Confirmed Local Unknown Example

| Sample ID    | Human Observation          | Taxonomy Status                | Recommended Action   |
| ------------ | -------------------------- | ------------------------------ | -------------------- |
| local_000001 | rubber slipper / flip-flop | outside_current_known_taxonomy | keep_as_unknown_test |

## Why This Matters for the Thesis

This stage supports the active-learning part of OpenWaste-HR.

The project is not only training a classifier. It is building a workflow where uncertain or unknown local samples can be reviewed, corrected, and later used to improve the dataset or update the model.

This connects the prototype to the research aim of hierarchical open-set waste classification with reject/manual-review support and local active learning.
