# Local Unknown Human Label Guide v1

## Purpose

This guide explains how to review local unknown images for OpenWaste-HR active learning v2.

## Main Rule

Do not force a local unknown object into a known class unless it clearly matches the current taxonomy.

If the object is not part of the known fine-label list, mark it as:

```text
outside_current_known_taxonomy
```

## Known Fine Labels

| Fine Label      | Meaning                                                          |
| --------------- | ---------------------------------------------------------------- |
| paper_cardboard | paper, cardboard, carton-like paper material                     |
| plastic         | plastic bottles, wrappers, containers, and similar plastic items |
| glass           | glass bottles, jars, or glass pieces                             |
| metal           | cans, metal containers, or visible metal waste                   |
| organic         | food waste, plant waste, or biodegradable organic material       |
| e_waste_battery | batteries or electronic waste; broader target taxonomy, not a current expanded-public retraining label |
| residual        | general non-recyclable waste already covered by the taxonomy     |

## Current Expanded-Public Review Rule

For the current expanded-public model and active learning v2 workflow, only these six labels should be treated as current known-class additions:

* paper_cardboard
* plastic
* glass
* metal
* organic
* residual

The broader project taxonomy still reserves `e_waste_battery`, but the current expanded-public training data does not support it as a known retraining class yet.

At this stage, batteries, chargers, cables, bulbs, and similar e-waste samples should stay as unknown-test or future-class candidates unless the project later adds dedicated trained support for that class.

## Recommended Human Fields

When reviewing a sample, record:

| Field                 | Meaning                                          |
| --------------------- | ------------------------------------------------ |
| sample_id             | image ID                                         |
| image_path            | local path to image                              |
| human_observed_object | what the object actually appears to be           |
| human_taxonomy_status | whether it is in or outside the current taxonomy |
| recommended_action    | what should happen to the sample                 |
| reviewer_notes        | short reason for the decision                    |

## Recommended Actions

| Action                        | Meaning                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| add_to_existing_class         | sample clearly belongs to a known fine label                 |
| keep_as_unknown_test          | sample is outside current taxonomy and should remain unknown |
| create_future_class_candidate | sample may justify a future new class                        |
| recollect_image               | image is unclear or low quality                              |
| exclude_duplicate             | duplicate or unusable sample                                 |
| manual_review                 | needs another reviewer or supervisor decision                |

## Example

| Sample ID    | Human Observation          | Taxonomy Status                | Recommended Action   |
| ------------ | -------------------------- | ------------------------------ | -------------------- |
| local_000001 | rubber slipper / flip-flop | outside_current_known_taxonomy | keep_as_unknown_test |

## Notes

A model prediction should not be treated as the final label.

The human reviewer should look at the image and decide based on the object itself, not only the predicted class or confidence score.
