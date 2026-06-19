# Confidence-Threshold Reject Baseline v1

## Purpose

This stage adds the first reject-option baseline to OpenWaste-HR.

The previous baseline was a closed-set forced-choice classifier. It always predicted one known fine label, even when confidence was low.

This confidence-threshold baseline allows the system to reject uncertain predictions and return:

```text
manual_review