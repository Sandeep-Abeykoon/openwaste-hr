# Local Unknown Dataset Protocol v1

## Purpose

This protocol defines how OpenWaste-HR collects and records local unknown or difficult waste images.

The aim is to create a small local evaluation dataset that tests whether the system can send uncertain, ambiguous, shifted, or locally unusual waste items to manual review instead of forcing a wrong known label.

## Why This Dataset Is Needed

The first TrashNet baseline only evaluates known classes.

However, the OpenWaste-HR research problem is about open-world reliability. Real waste streams may include objects that are:

- unknown to the training label space
- mixed-material
- damaged or deformed
- contaminated
- locally specific
- visually ambiguous
- photographed under different lighting or backgrounds

Therefore, a local unknown dataset is required before claiming unknown/manual-review performance.

## Safety Rule

Only photograph safe household or public waste items.

Do not handle dangerous waste directly. Do not break glass, open batteries, touch sharp objects, open medicine containers, or handle chemical containers. If an item looks unsafe, skip it.

## Collection Target

Initial target:

| Type | Target Count |
|---|---:|
| local unknown / difficult images | 30 to 50 |
| minimum for first test | 20 |

Later target:

| Type | Target Count |
|---|---:|
| stronger local unknown dataset | 100+ |

## Image Capture Rules

For each image:

1. Use a phone camera.
2. Capture one main waste item if possible.
3. Include some difficult cases, but keep them safe.
4. Use different backgrounds.
5. Use different lighting conditions.
6. Avoid showing faces, people, addresses, or private information.
7. Do not edit images heavily.

## Good Unknown/Difficult Examples

| Example | Why useful |
|---|---|
| shiny food wrapper | reflective mixed-material packaging |
| dirty plastic container | plastic vs residual ambiguity |
| crushed packaging | deformation |
| local Sri Lankan packaging | local domain shift |
| multi-layer snack packet | mixed-material uncertainty |
| stained paper packaging | paper/cardboard vs residual ambiguity |
| unusual cable/charger | possible e-waste-like item |
| unclear object under poor lighting | low-confidence/manual-review case |

## Labels

For the first unknown evaluation, these images are not treated as known training classes.

They use:

| Field | Value |
|---|---|
| source_dataset | local_phone_images |
| original_label | unknown |
| fine_label | unknown |
| coarse_label | unknown |
| is_known | false |

## Usage Values

Allowed usage values:

| Usage | Meaning |
|---|---|
| unknown_test | used for unknown/manual-review evaluation |
| local_unknown | local unknown dataset storage |
| active_learning_candidate | later human-correction/adaptation candidate |

## Manifest Output

The manifest builder creates:

```text
ml/data/splits/local_unknown_manifest_v1.csv