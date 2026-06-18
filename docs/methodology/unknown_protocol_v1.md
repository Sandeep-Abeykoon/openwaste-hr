# Unknown-Class Protocol v1

## Purpose

This protocol defines how OpenWaste-HR will evaluate unknown, out-of-distribution, ambiguous, or locally unusual waste items.

The aim is not only to measure classification accuracy, but to test whether the system can avoid confident wrong predictions.

## Known Classes

The known fine classes used for model training are:

1. paper_cardboard
2. plastic
3. glass
4. metal
5. organic
6. e_waste_battery
7. residual

## Unknown Items

Unknown items are images that should not be forced into one of the known fine labels.

Unknown examples may include:

- mixed-material packaging
- damaged or deformed objects
- dirty or contaminated objects
- reflective wrappers
- local/regional packaging not represented in the training set
- objects from excluded waste categories
- unclear images where the object is not visually identifiable
- images with multiple overlapping waste items

## Evaluation Unknown Types

### Type A — Held-out unknown subclasses

Some subclasses will be deliberately excluded from training and used only during testing.

Example candidates:

- textile waste
- wood waste
- ceramic waste
- rubber waste
- medicine packaging
- unusual electronic accessories

### Type B — Local unknown images

The project will include a small local dataset captured using a phone camera.

Local unknown images should include:

- Sri Lankan/local packaging
- mixed-material food packaging
- shiny wrappers
- dirty plastic or paper items
- damaged bottles or crushed cans
- ambiguous objects
- objects photographed under different lighting

### Type C — Cross-domain stress test images

Images from a different dataset or source may be used to test whether the model generalises outside the main training distribution.

## Training Rule

Unknown images must not be included as normal training labels in the first closed-set baseline.

## Evaluation Rule

Unknown images are used to measure:

- whether the model rejects unknown inputs
- whether the model avoids confident wrong predictions
- whether the model falls back safely to a coarse category
- whether local active learning improves later performance

## Active Learning Rule

Rejected or uncertain local images should be saved for human correction.

The corrected samples can later be used in a small fine-tuning experiment to test local adaptation.