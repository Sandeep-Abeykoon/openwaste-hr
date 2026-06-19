# Hierarchical Decision Policy v1

## Purpose

This stage implements the first OpenWaste-HR hierarchical decision policy.

The previous baselines made only two decisions:

1. accept fine-label prediction
2. manual review

This stage adds a middle decision:

1. fine label
2. coarse label
3. manual review

## Motivation

A normal closed-set classifier always forces every image into a known fine label. This is unsafe for unknown, ambiguous, mixed-material, or locally unusual waste items.

The hierarchical decision policy makes the system less brittle.

Instead of forcing a fine label when the model is uncertain, it can fall back to a broader coarse category. If even the coarse evidence is not stable, the system sends the image to manual review.

## Decision Outputs

| Output Type   | Meaning                                                                             |
| ------------- | ----------------------------------------------------------------------------------- |
| fine_label    | The model is confident enough to return a detailed class such as plastic or metal   |
| coarse_label  | The model is not confident enough for a fine class, but the broader group is stable |
| manual_review | The model should not make a safe automated decision                                 |

## Decision Rule

For each image:

1. Read the model fine-label probabilities.
2. Compute the maximum fine-label confidence.
3. Aggregate fine-label probabilities into coarse-label probabilities.
4. Apply the decision policy.

## Policy v1

| Condition                                                                       | Output        |
| ------------------------------------------------------------------------------- | ------------- |
| fine confidence is high                                                         | fine label    |
| fine confidence is not high, but coarse confidence and coarse margin are stable | coarse label  |
| otherwise                                                                       | manual review |

## Current Coarse Mapping

| Fine Label      | Coarse Label |
| --------------- | ------------ |
| paper_cardboard | recyclable   |
| plastic         | recyclable   |
| glass           | recyclable   |
| metal           | recyclable   |
| residual        | residual     |

## Thresholds

Policy v1 uses fixed thresholds selected from previous validation and baseline experiments:

| Threshold                     | Purpose                                                                   |
| ----------------------------- | ------------------------------------------------------------------------- |
| fine_confidence_threshold     | minimum confidence for fine-label output                                  |
| coarse_confidence_threshold   | minimum aggregated confidence for coarse fallback                         |
| coarse_margin_threshold       | minimum separation between top coarse category and second coarse category |
| minimum_confidence_for_coarse | prevents very low-confidence images from receiving a coarse label         |

## Research Meaning

This stage is the first implementation of the main OpenWaste-HR decision policy.

It directly supports the project aim of moving beyond forced closed-set classification. The system can now produce detailed predictions, broader fallback predictions, or manual-review warnings.
