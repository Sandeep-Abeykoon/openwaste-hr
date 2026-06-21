# Reviewed Local Label Seed Summary v1

## Purpose

This stage creates the first reviewed local-label seed for active learning v2.

## Confirmed Sample

| Field              | Value                          |
| ------------------ | ------------------------------ |
| sample_id          | local_000001                   |
| human observation  | rubber slipper / flip-flop     |
| taxonomy status    | outside_current_known_taxonomy |
| recommended action | keep_as_unknown_test           |
| proposed new label | rubber_slipper_flip_flop       |

## Why This Matters

The best current model predicted a known class/coarse category for this sample, even though it is outside the current known taxonomy.

This gives a useful thesis example showing that high confidence does not guarantee correct open-world recognition.

## Active Learning Link

The reviewed seed supports the human-in-the-loop active learning part of OpenWaste-HR.

It can later be used as:

| Role                          | Meaning                                   |
| ----------------------------- | ----------------------------------------- |
| keep_as_unknown_test          | keep the image for open-set evaluation    |
| create_future_class_candidate | use it as a possible future class example |

## Next Stage

The next stage should prepare the active-learning v2 dataset plan using the reviewed local-label seed.
