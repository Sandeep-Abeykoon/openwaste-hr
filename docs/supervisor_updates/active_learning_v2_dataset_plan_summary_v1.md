# Active Learning v2 Dataset Plan Summary v1

## Purpose

This stage creates the first dataset plan for active learning v2.

## Input

```text id="zuxhz3"
ml/data/splits/reviewed_local_labels_seed_v1.csv
```

## Reviewed Seed Used

| Field              | Value                          |
| ------------------ | ------------------------------ |
| sample_id          | local_000001                   |
| human observation  | rubber slipper / flip-flop     |
| taxonomy status    | outside_current_known_taxonomy |
| recommended action | keep_as_unknown_test           |
| proposed new label | rubber_slipper_flip_flop       |

## Dataset Decision

The reviewed sample should not be added to existing known training classes.

Instead, it should be used as:

| Role                   | Value |
| ---------------------- | ----- |
| unknown-test sample    | yes   |
| future class candidate | yes   |
| known training sample  | no    |

## Why This Matters

The plan prevents the system from forcing real local unknown objects into existing labels.

This supports the OpenWaste-HR goal of building a safer waste classification pipeline with human-in-the-loop active learning.

## Next Stage

After this, the project can prepare an active learning v2 thesis section and later expand the reviewed local dataset when more human labels are available.
