# Open-Set Score Baseline v1

## Purpose

This stage adds two logit-based reject-option baselines to OpenWaste-HR:

1. Maximum Logit Score
2. Energy Score

The previous confidence-threshold baseline used maximum softmax confidence. Softmax confidence can be overconfident, so this stage records scores directly from model logits.

## Methods

### Maximum Logit Score

The Maximum Logit Score is:

```text
max_logit_score = max(logits)