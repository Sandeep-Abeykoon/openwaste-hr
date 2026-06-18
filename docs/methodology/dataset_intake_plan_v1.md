# OpenWaste-HR Dataset Intake Plan v1

## Purpose

This document defines how datasets will be collected, stored, mapped, and evaluated in OpenWaste-HR.

The project does not treat all waste images as one simple dataset. Instead, each image must be tracked with its source dataset, original label, mapped OpenWaste-HR fine label, mapped coarse label, and intended experiment role.

This is important because waste datasets often use inconsistent categories, backgrounds, image conditions, and annotation formats.

## Planned Dataset Sources

| Source | Planned Role | Notes |
|---|---|---|
| TrashNet | Simple baseline dataset | Useful for early known-class model development |
| GlobalWasteData | Main dataset candidate | Useful because it addresses fragmented public waste datasets |
| TACO | Cross-domain stress testing | Useful for in-the-wild litter, clutter, and difficult scenes |
| Local phone images | Unknown testing and active learning | Used to test local Sri Lankan waste conditions |

## Dataset Storage Rule

Large image files must not be committed to GitHub.

Images should be stored locally under:

- `ml/data/raw/`
- `ml/data/interim/`
- `ml/data/processed/`
- `ml/data/local_unknown/`

Only small CSV files, mapping files, split files, and documentation should be committed.

## Manifest Rule

Every image used in the project must appear in a manifest CSV file.

The manifest must record:

- sample ID
- source dataset
- original label
- mapped fine label
- mapped coarse label
- known/unknown status
- usage role
- relative image path
- license or citation note

## Known Training Data

Known training data can only use these OpenWaste-HR fine labels:

1. paper_cardboard
2. plastic
3. glass
4. metal
5. organic
6. e_waste_battery
7. residual

## Unknown Evaluation Data

Unknown images are not trained as a normal fine class in the first baseline.

Unknown images are used for:

- unknown detection evaluation
- reject-option testing
- manual review testing
- local active learning experiments

## Dataset Intake Order

1. Start with a simple baseline source such as TrashNet.
2. Map original dataset labels into the OpenWaste-HR taxonomy.
3. Create a small manifest CSV.
4. Verify the manifest using automated tests.
5. Train the first known-class baseline.
6. Add unknown and local images only after the known-class pipeline works.
7. Use TACO and local images for harder stress testing.
8. Use GlobalWasteData as the stronger dataset source when the baseline pipeline is stable.

## Research Safety Rule

Do not claim novelty from simply using a public dataset.

The novelty comes from:

- hierarchical coarse/fine decision output
- reject option for unknown items
- open-world evaluation
- local active learning and correction loop
- safer fallback behaviour compared with forced closed-set prediction