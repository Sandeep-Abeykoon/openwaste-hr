# Local Unknown Dataset Protocol v1

## Purpose

This protocol defines the corrected local unknown dataset for OpenWaste-HR.

The local unknown dataset must contain images that should not be confidently classified into the current closed-set baseline labels.

The current TrashNet baseline model is trained only on:

1. paper_cardboard
2. plastic
3. glass
4. metal
5. residual

Therefore, the local unknown dataset should contain items that are outside these trained labels, visually ambiguous, mixed-material, or unsafe to classify confidently from an image.

## What Should Be Captured

Capture safe household or local objects that are not clear examples of the trained classes.

Good examples:

| Example | Reason |
|---|---|
| textile / cloth waste | outside current trained labels |
| wood pieces | outside current trained labels |
| ceramic item | outside current trained labels |
| rubber item | outside current trained labels |
| sponge / foam item | outside current trained labels |
| shoe/slipper piece | outside current trained labels |
| mixed-material laminated packet | difficult/ambiguous |
| dirty mixed-material item | difficult to classify safely |
| crushed or deformed object | visual distribution shift |
| unclear local packaging | manual-review candidate |

## What Should Not Be Used in This Dataset

Do not include clear examples of the current trained classes.

Avoid using:

| Avoid | Reason |
|---|---|
| clear plastic bottle | belongs to plastic |
| clear metal can | belongs to metal |
| clear glass bottle or jar | belongs to glass |
| clear paper/cardboard | belongs to paper_cardboard |
| normal TrashNet-like trash wrapper | may belong to residual |

These images can be used later for a separate local known challenge dataset, but not for the pure unknown/manual-review evaluation.

## Ambiguous Items

Some items may contain plastic, metal, or paper but still be difficult because they are mixed, dirty, reflective, damaged, or unclear.

These can be included only if the intended behaviour is manual review.

Example:

| Item | Include? | Reason |
|---|---|---|
| shiny mixed snack packet | yes | mixed/reflective/manual-review |
| dirty food packaging | yes | contamination makes class uncertain |
| clear clean plastic bottle | no | known plastic class |
| clean aluminium can | no | known metal class |

## Safety Rule

Only photograph safe items.

Do not handle broken glass, sharp objects, leaking batteries, chemicals, medicine waste, or anything unsafe. If an item looks unsafe, skip it.

## Collection Target

Minimum for first corrected evaluation:

| Dataset | Target |
|---|---:|
| corrected local unknown images | 30 to 50 |

Recommended first target:

| Dataset | Target |
|---|---:|
| corrected local unknown images | 40 |

## Manifest Rule

Every image in this corrected dataset should use:

| Field | Value |
|---|---|
| source_dataset | local_phone_images |
| original_label | unknown |
| fine_label | unknown |
| coarse_label | unknown |
| is_known | false |
| usage | unknown_test |

## Research Note

This corrected dataset is used only for unknown/manual-review evaluation.

It must not be used to train the first closed-set baseline.