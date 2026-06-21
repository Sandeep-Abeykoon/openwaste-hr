# RealWaste Intake v1

## Purpose

This stage adds the first public dataset expansion pathway for OpenWaste-HR.

The current v1 model was trained mainly using the TrashNet-style dataset. RealWaste is introduced as the next dataset because it is closer to real landfill waste conditions and is suitable for image-classification style training.

## Dataset

The dataset targeted in this stage is:

```text
RealWaste
```

Expected local root:

```text
ml/data/raw/realwaste
```

The expected class folders are:

| RealWaste Label     | Planned OpenWaste-HR Mapping          |
| ------------------- | ------------------------------------- |
| Cardboard           | paper_cardboard                       |
| Food Organics       | organic                               |
| Glass               | glass                                 |
| Metal               | metal                                 |
| Miscellaneous Trash | residual                              |
| Paper               | paper_cardboard                       |
| Plastic             | plastic                               |
| Textile Trash       | future_class_candidate / unknown_test |
| Vegetation          | organic                               |

## Why RealWaste Is Useful

RealWaste is useful because it extends the project beyond the small TrashNet-style dataset.

The goal is to test whether a more realistic public dataset improves:

* known-class accuracy
* macro-F1
* local unknown handling
* hierarchical decision quality
* safe policy reliability

## Safe Mapping Rule

The most important rule is:

```text
Do not force outside-taxonomy labels into current known classes.
```

For this reason, Textile Trash is not mapped into residual or plastic. It is kept as an unknown-test and future-class candidate.

## Output Manifests

This stage creates:

| Output                        | Purpose                                       |
| ----------------------------- | --------------------------------------------- |
| realwaste_manifest_v1.csv     | all included RealWaste samples                |
| realwaste_known_train_v1.csv  | known training split                          |
| realwaste_known_val_v1.csv    | known validation split                        |
| realwaste_known_test_v1.csv   | known test split                              |
| realwaste_unknown_test_v1.csv | outside-taxonomy unknown/future-class samples |

## Research Meaning

This stage prepares Baseline C:

```text
Baseline C = pretrained expanded public dataset model
```

After the manifest is built and inspected, the next phase can train a pretrained model using TrashNet plus RealWaste known samples and compare it against the current best pretrained safe hierarchical policy.
