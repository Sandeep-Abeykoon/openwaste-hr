# Public Dataset Expansion Plan v1

## Purpose

This stage prepares the next research phase for OpenWaste-HR.

The current working system was trained and evaluated mainly using the TrashNet-style dataset. This created a complete v1 pipeline, but it is not enough for the final strongest research version because the dataset is small and limited.

The next phase should expand the training and evaluation data using additional public waste datasets.

## Why Dataset Expansion Is Needed

The current model achieved strong results after pretrained training and safe hierarchical tuning, but it is still limited by the training data.

A small closed-set dataset can cause the model to produce high confidence for unfamiliar local objects. The local unknown example local_000001 showed this clearly: a rubber slipper / flip-flop was predicted as a known class with high confidence.

Therefore, the next research phase should test whether adding larger and more realistic public datasets improves:

* known-class accuracy
* macro-F1
* robustness to varied backgrounds
* local unknown handling
* hierarchical decision quality
* final OpenWaste-HR policy reliability

## Current Baseline

The current completed model sequence is:

| Model      | Training Data                                              | Status   |
| ---------- | ---------------------------------------------------------- | -------- |
| Baseline A | TrashNet-style dataset, scratch training                   | complete |
| Baseline B | TrashNet-style dataset, pretrained transfer learning       | complete |
| Baseline C | expanded public dataset, pretrained transfer learning      | planned  |
| Baseline D | expanded public dataset plus reviewed active-learning data | planned  |

## Candidate Public Datasets

The next dataset expansion should focus on image-classification-compatible datasets first.

| Priority | Dataset   | Reason                                                                                                                               |
| -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1        | RealWaste | suitable next dataset because it is closer to image classification and real-world waste conditions                                   |
| 2        | TACO      | useful later for in-the-wild waste and hierarchical/unknown evaluation, but more complex because it is annotation/detection oriented |

## Recommended Next Dataset

The recommended next dataset is:

```text id="kcwj3t"
RealWaste
```

Reason:

RealWaste is a better immediate next step because it is closer to the current image-classification training pipeline. It should be easier to map into the existing OpenWaste-HR fine/coarse taxonomy than an object-detection or segmentation dataset.

## Future Dataset Candidate

TACO should be considered after the image-classification expansion stage.

TACO is useful because it contains waste in more natural environments and has a hierarchical annotation structure. However, it may require additional preprocessing because the current OpenWaste-HR training pipeline expects one image-level label per sample.

## Mapping Principle

The most important rule is:

```text id="uzs81c"
Do not force unclear or outside-taxonomy labels into known classes.
```

Each new dataset label should be mapped into one of the following roles:

| Mapping Role           | Meaning                                         |
| ---------------------- | ----------------------------------------------- |
| known_train_candidate  | clear mapping to current known fine label       |
| known_eval_candidate   | clear mapping but reserved for testing          |
| unknown_eval_candidate | outside current known taxonomy                  |
| future_class_candidate | useful possible future class                    |
| exclude_or_review      | unclear, ambiguous, mixed, or low-quality label |

## Current OpenWaste-HR Fine Labels

| Fine Label      | Coarse Label |
| --------------- | ------------ |
| paper_cardboard | recyclable   |
| plastic         | recyclable   |
| glass           | recyclable   |
| metal           | recyclable   |
| organic         | organic      |
| e_waste_battery | hazardous    |
| residual        | residual     |

## Expanded Dataset Training Plan

After the next dataset is downloaded and inspected, the workflow should be:

1. inspect folder/label names
2. create source label mapping CSV
3. build expanded manifest
4. validate manifest
5. train pretrained expanded model
6. evaluate closed-set performance
7. evaluate reject-option performance
8. evaluate local unknown behaviour
9. tune safe hierarchical policy again
10. compare against current best policy

## Planned Comparison

The final comparison should include:

| System                                                           | Purpose                                |
| ---------------------------------------------------------------- | -------------------------------------- |
| Baseline A: scratch TrashNet                                     | weak baseline                          |
| Baseline B: pretrained TrashNet                                  | current strong baseline                |
| Baseline C: pretrained expanded dataset                          | tests public dataset expansion         |
| Baseline D: expanded dataset plus reviewed local active learning | tests human feedback improvement       |
| Final OpenWaste-HR policy                                        | best selected safe hierarchical policy |

## Expected Research Value

This phase will make the thesis stronger because it will show whether the system improves when trained with more diverse public waste data.

It will also make the evaluation more convincing because the project will no longer rely only on the small TrashNet-style dataset.

## Next Step

The next implementation step should be to create a RealWaste intake plan and manifest builder.

The goal is not to train immediately. The goal is first to inspect the dataset structure and create a safe label-mapping process.
