# Expanded Public Local and Public Unknown Evaluation v1

## Purpose

This stage evaluates unknown-handling behaviour for Baseline C.

Baseline C is the pretrained expanded public dataset model trained using combined TrashNet-style and RealWaste known samples.

## Baseline C Definition

```text id="ze7urw"
Baseline C = pretrained expanded public dataset model
```

## Evaluation Goal

The goal is to test whether the expanded public pretrained model improves unknown handling compared with the earlier TrashNet-only pretrained model.

This stage evaluates two unknown sources:

| Unknown Source                    | Purpose                                                             |
| --------------------------------- | ------------------------------------------------------------------- |
| local unknown dataset             | tests real local/outside-taxonomy images collected for OpenWaste-HR |
| public unknown/future-class split | tests RealWaste Textile Trash as public outside-taxonomy data       |

## Input Model

| Item          | Path                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| checkpoint    | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_best.pt            |
| class mapping | ml/outputs/checkpoints/expanded_public_pretrained_v1/expanded_public_pretrained_v1_class_mapping.json |

## Unknown Evaluation Inputs

| Evaluation                | Manifest                                           |
| ------------------------- | -------------------------------------------------- |
| local unknown evaluation  | ml/data/splits/local_unknown_manifest_v1.csv       |
| public unknown evaluation | ml/data/splits/expanded_public_unknown_test_v1.csv |

## Public Unknown Meaning

The public unknown/future-class split comes from RealWaste:

```text id="sahrl3"
Textile Trash
```

This class is intentionally mapped as:

| Field        | Value                  |
| ------------ | ---------------------- |
| fine_label   | unknown                |
| coarse_label | unknown                |
| is_known     | false                  |
| usage        | unknown_test           |
| mapping_role | future_class_candidate |

## Methods Evaluated

This stage evaluates the same unknown rejection methods used earlier:

| Method               | Description                                                             |
| -------------------- | ----------------------------------------------------------------------- |
| confidence threshold | rejects unknown samples when confidence is below the selected threshold |
| max-logit score      | rejects unknown samples using the selected max-logit threshold          |
| energy score         | rejects unknown samples using the selected energy threshold             |

## Metrics

The evaluation reports:

* unknown rejection count
* unknown accepted count
* unknown rejection rate
* false acceptance rate
* accepted predicted-label distribution

## Research Meaning

This is a key stage because the expanded public model should be tested beyond closed-set accuracy.

The project needs to show whether dataset expansion improves:

* known-class classification
* selective prediction
* local unknown rejection
* public unknown/future-class rejection

## Next Stage

After this stage, the next step should tune a safe hierarchical policy for the expanded public pretrained model.
