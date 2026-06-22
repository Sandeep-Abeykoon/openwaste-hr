# Manual Review Records Completion Summary v1

## Purpose

This stage defines how manual review records should be completed before active learning retraining.

## Reason

The project includes local active learning, but the full retraining cycle should not be performed until reviewed samples are safely separated into known-class samples and unknown/future-class samples.

## Key Rule

Reviewed samples should only be added to known training if they clearly belong to the current known taxonomy.

Known fine labels:

* paper_cardboard
* plastic
* glass
* metal
* organic
* residual

## Example

The reviewed local sample `local_000001` was observed as a rubber slipper / flip-flop.

This is outside the current known taxonomy, so it should not be used for known-class retraining.

It should remain an unknown/future-class candidate.

## Next Stage

The next stage is to complete or audit the manual review records, then count how many reviewed samples are valid known-class retraining candidates.

If enough known-class samples exist, the project can continue with active learning v2 retraining.

If not, the project should report active learning as a workflow preparation and future improvement stage.
