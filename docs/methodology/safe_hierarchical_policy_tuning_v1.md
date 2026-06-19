# Safe Hierarchical Policy Tuning v1

## Purpose

This stage tunes the OpenWaste-HR hierarchical decision policy.

The previous policy introduced three outputs:

1. fine label
2. coarse label
3. manual review

However, the first policy was too permissive for local unknown images. It allowed too many local unknown images to receive fine-label or coarse-label outputs instead of being routed to manual review.

## Motivation

OpenWaste-HR prioritises safe handling of unknown, ambiguous, mixed-material, damaged, contaminated, and locally unusual images.

A useful hierarchical policy should not only perform well on known test samples. It should also reduce unsafe acceptance of local unknown samples.

## Tuning Goal

The tuning process searches across threshold combinations for:

| Threshold                     | Meaning                                                                   |
| ----------------------------- | ------------------------------------------------------------------------- |
| fine_confidence_threshold     | minimum confidence required for fine-label output                         |
| coarse_confidence_threshold   | minimum aggregated coarse confidence required for coarse fallback         |
| coarse_margin_threshold       | minimum separation between top coarse category and second coarse category |
| minimum_confidence_for_coarse | minimum fine confidence required before allowing coarse fallback          |

## Selection Logic

Each candidate policy is evaluated on:

1. known-test decision coverage
2. known-test hierarchical success rate
3. local unknown manual-review rate
4. local unknown acceptance rate

The selected policy should preserve useful known-class decisions while improving manual-review routing for local unknown images.

## Research Meaning

This stage improves the hierarchical decision logic by making coarse fallback safer.

The system should only use coarse fallback when the broader category is stable enough. Otherwise, it should send the image to manual review.
