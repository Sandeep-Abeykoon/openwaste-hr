# TrashNet Intake v1

## Purpose

TrashNet is used as the first simple closed-set dataset source for OpenWaste-HR.

It is not used as the main novelty of the project. It is used to build and verify the first known-class training pipeline before adding harder unknown, cross-domain, and local active-learning data.

## Why TrashNet Is Used First

TrashNet is suitable for the first baseline because it uses simple folder-per-class image labels:

- cardboard
- paper
- plastic
- glass
- metal
- trash

These labels can be mapped cleanly into the OpenWaste-HR taxonomy.

## OpenWaste-HR Mapping

| TrashNet Label | OpenWaste-HR Fine Label | OpenWaste-HR Coarse Label |
|---|---|---|
| cardboard | paper_cardboard | recyclable |
| paper | paper_cardboard | recyclable |
| plastic | plastic | recyclable |
| glass | glass | recyclable |
| metal | metal | recyclable |
| trash | residual | residual |

## Important Limitation

TrashNet images are relatively clean and simple compared with real-world waste streams. Therefore, TrashNet will only be used for early closed-set baseline development.

Later project stages must include:

- unknown test images
- local phone-camera images
- cross-domain stress tests
- reject-option evaluation

## Expected Local Folder Structure

After downloading and extracting TrashNet, the local folder should look like this:

```text
ml/data/raw/trashnet/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/