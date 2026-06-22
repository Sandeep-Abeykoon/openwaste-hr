# Final Expanded Public Model and Policy Comparison v1

## Purpose

This document compares the main OpenWaste-HR model and policy results after adding RealWaste to the original TrashNet-based workflow.

The comparison focuses on:

* closed-set known classification
* selective prediction with reject option
* local unknown handling
* public unknown/future-class handling
* safe hierarchical decision policy performance

## Compared Systems

| System        | Description                               |
| ------------- | ----------------------------------------- |
| Baseline A    | Scratch TrashNet-only CNN baseline        |
| Baseline B    | Pretrained TrashNet-only model            |
| Baseline C    | Pretrained expanded public dataset model  |
| Safe Policy B | Safe hierarchical policy using Baseline B |
| Safe Policy C | Safe hierarchical policy using Baseline C |

## Closed-Set Known Classification

| Model      | Known Test Set             | Classes | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---------- | -------------------------- | ------: | -------: | ----------------: | -------: | ----------: |
| Baseline A | TrashNet known test        |       5 |   0.6927 |            0.6545 |   0.6456 |      0.7009 |
| Baseline B | TrashNet known test        |       5 |   0.8880 |            0.8431 |   0.8510 |      0.8873 |
| Baseline C | Expanded public known test |       6 |   0.8876 |            0.8750 |   0.8819 |      0.8870 |

## Closed-Set Interpretation

The expanded public pretrained model achieved similar overall accuracy to the TrashNet-only pretrained model but improved macro-F1.

This matters because macro-F1 is more sensitive to class-balanced performance. The expanded public model also includes the organic class, which was missing from the original TrashNet-only training setup.

## Expanded Public Reject-Option Results

| Method               | Coverage | Rejection Rate | Selective Accuracy | Selective Macro-F1 | Selective Weighted-F1 |
| -------------------- | -------: | -------------: | -----------------: | -----------------: | --------------------: |
| Confidence threshold |   0.7229 |         0.2771 |             0.9789 |             0.9732 |                0.9788 |
| Max-logit            |   0.7362 |         0.2638 |             0.9677 |             0.9627 |                0.9676 |
| Energy               |   0.7181 |         0.2819 |             0.9668 |             0.9612 |                0.9668 |

## Reject-Option Interpretation

For known selective prediction, the confidence-threshold method performed best because it achieved the highest selective macro-F1 and selective weighted-F1.

This shows that rejecting uncertain known-test samples greatly improves the reliability of accepted predictions.

## Expanded Public Unknown Handling

### Local Unknown Dataset

| Method     | Unknown Samples | Rejected | Accepted as Known | Unknown Rejection Rate | False Acceptance Rate |
| ---------- | --------------: | -------: | ----------------: | ---------------------: | --------------------: |
| Confidence |              40 |       24 |                16 |                 0.6000 |                0.4000 |
| Max-logit  |              40 |       26 |                14 |                 0.6500 |                0.3500 |
| Energy     |              40 |       27 |                13 |                 0.6750 |                0.3250 |

### Public Unknown/Future-Class Split

| Method     | Unknown Samples | Rejected | Accepted as Known | Unknown Rejection Rate | False Acceptance Rate |
| ---------- | --------------: | -------: | ----------------: | ---------------------: | --------------------: |
| Confidence |             318 |      200 |               118 |                 0.6289 |                0.3711 |
| Max-logit  |             318 |      202 |               116 |                 0.6352 |                0.3648 |
| Energy     |             318 |      207 |               111 |                 0.6509 |                0.3491 |

## Unknown Handling Interpretation

Energy-score rejection was the strongest standalone unknown-rejection method for the expanded public model.

This is important because Step 57 showed confidence threshold was strongest for selective known-class prediction, while Step 58 showed energy score was strongest for rejecting unknown samples.

Therefore, known-class reliability and unknown rejection do not always favour the same uncertainty method.

## Safe Hierarchical Policy Comparison

| Policy        | Known Coverage | Known Manual Review Rate | Accepted Success Rate | Local Unknown Manual Review Rate | Local Unknown Acceptance Rate |
| ------------- | -------------: | -----------------------: | --------------------: | -------------------------------: | ----------------------------: |
| Safe Policy B |         0.8646 |                   0.1354 |                0.9608 |                           0.6000 |                        0.4000 |
| Safe Policy C |         0.8819 |                   0.1181 |                0.9838 |                           0.4750 |                        0.5250 |

## Safe Policy Interpretation

Safe Policy C improved known-class usefulness and accepted-decision reliability:

* higher known decision coverage
* lower known manual review rate
* higher accepted hierarchical success rate

However, Safe Policy B was stricter on the local unknown dataset because it sent more local unknown samples to manual review.

This means Safe Policy C is better for balanced known-class usability and accepted-decision correctness, while Safe Policy B is more conservative for local unknown rejection.

## Final Research Position

The final OpenWaste-HR result should be presented as a trade-off:

| Finding                                                                  | Meaning                                                   |
| ------------------------------------------------------------------------ | --------------------------------------------------------- |
| Expanded public training improved macro-F1                               | broader dataset improved class-balanced known performance |
| Confidence threshold performed best for known selective prediction       | best accepted known prediction reliability                |
| Energy score performed best for unknown rejection                        | strongest standalone unknown detector                     |
| Expanded safe hierarchical policy improved accepted-decision reliability | better final balanced policy for known usability          |
| Earlier safe policy remained stricter for local unknown rejection        | useful limitation and future improvement direction        |

## Recommended Final Thesis Claim

The strongest final system is the expanded public pretrained model with safe hierarchical decision-making, because it combines broader known-class training, high accepted-decision reliability, and hierarchical fallback.

However, the thesis should also clearly state that energy-score rejection produced stronger standalone unknown rejection. A future version of OpenWaste-HR could combine the expanded safe hierarchical policy with an additional energy-based unknown safety gate.
