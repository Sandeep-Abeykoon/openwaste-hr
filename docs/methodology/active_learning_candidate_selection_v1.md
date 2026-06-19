# Active Learning Candidate Selection v1

## Purpose

This stage selects local unknown dataset images that should be sent for human labelling.

The goal is to support the active learning part of OpenWaste-HR. Instead of randomly choosing images, the system prioritises images that are uncertain, manually reviewed, or suspiciously accepted as known labels.

## Why Active Learning Is Needed

The current model is trained mainly on known TrashNet-derived categories. Local phone-captured images can contain unknown, mixed, damaged, contaminated, or locally unusual items.

When the model is unsure, or when it confidently accepts a local unknown image as a known label, the sample becomes useful for human review.

Human-labelled samples can later be used to:

1. expand the dataset
2. improve threshold tuning
3. improve open-set/manual-review behaviour
4. identify new local waste categories

## Candidate Types

| Candidate Type                    | Meaning                                                                                   |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| manual_review_candidate           | the tuned policy already routed this image to manual review                               |
| coarse_fallback_unknown_candidate | the image received a coarse label even though it belongs to the local unknown dataset     |
| fine_accepted_unknown_candidate   | the image received a fine known label even though it belongs to the local unknown dataset |

## Ranking Logic

Each sample receives an active learning score using:

| Factor                    | Meaning                                                       |
| ------------------------- | ------------------------------------------------------------- |
| decision priority         | manual-review and suspicious accepted samples are prioritised |
| prediction entropy        | higher entropy means more uncertainty                         |
| confidence uncertainty    | lower confidence means more uncertainty                       |
| coarse margin uncertainty | smaller margin means the coarse decision is less stable       |

## Output

The output is a ranked CSV file of candidate images for human labelling.

Each selected candidate includes:

* sample ID
* image path
* object description
* model prediction
* hierarchical decision type
* confidence values
* active learning score
* recommended human action

## Research Meaning

This step connects the reject-aware classifier to the local active learning loop.

The model does not only make predictions; it also identifies which local cases should be reviewed by a human next.
