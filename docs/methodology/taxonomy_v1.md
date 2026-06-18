# OpenWaste-HR Taxonomy v1

## Purpose

This file defines the first frozen taxonomy for OpenWaste-HR.

The project uses a hierarchical label structure because real waste datasets often contain inconsistent labels across sources. A fixed taxonomy is required before dataset harmonisation, baseline training, open-set evaluation, and active learning can be implemented.

## Coarse Labels

| Coarse ID | Coarse Label | Meaning |
|---:|---|---|
| 0 | recyclable | Common recyclable materials |
| 1 | organic | Biodegradable food or plant-based waste |
| 2 | hazardous | Waste needing special handling |
| 3 | residual | General non-recyclable or unclear everyday waste |

## Fine Labels

| Fine ID | Fine Label | Coarse Label | Example Items |
|---:|---|---|---|
| 0 | paper_cardboard | recyclable | paper, cardboard, paper packaging |
| 1 | plastic | recyclable | plastic bottle, plastic cup, plastic container |
| 2 | glass | recyclable | glass bottle, glass jar |
| 3 | metal | recyclable | aluminium can, tin, metal lid |
| 4 | organic | organic | food waste, fruit peel, vegetable waste |
| 5 | e_waste_battery | hazardous | battery, charger, cable, small e-waste |
| 6 | residual | residual | tissue, dirty wrapper, contaminated packaging |

## Reserved Labels

| Label | Purpose |
|---|---|
| unknown | Used for open-set rejection and evaluation only |
| manual_review | Used as a user-facing output when the system should not decide confidently |

## Important Research Rule

Unknown items must not be trained as a normal known class in the first baseline model. The unknown label is used to test whether the model can reject unfamiliar, ambiguous, damaged, mixed, or locally unusual waste items.

## System Output Types

The system can produce three types of output:

1. Fine-label prediction  
   Example: `plastic`

2. Coarse fallback prediction  
   Example: `recyclable`, when the system is unsure whether the item is plastic, metal, glass, or paper/cardboard.

3. Unknown/manual review  
   Example: `manual_review`, when the item is too uncertain or appears outside the known label space.

## Version Decision

This is Taxonomy v1.

This taxonomy should not be changed casually after dataset preparation begins. If changes are needed later, create `taxonomy_v2.md` and explain the reason in the research log.