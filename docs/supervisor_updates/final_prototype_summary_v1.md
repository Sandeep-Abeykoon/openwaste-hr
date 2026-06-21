# OpenWaste-HR Supervisor Prototype Summary v1

## Current Status

OpenWaste-HR now has a working end-to-end prototype.

The system can process a waste image and return one of three decision types:

| Decision Type | Meaning                                       |
| ------------- | --------------------------------------------- |
| fine_label    | detailed known waste label                    |
| coarse_label  | broader fallback category                     |
| manual_review | uncertain or unsafe case sent to human review |

## Working Demo Flow

```text
frontend demo page → FastAPI backend → API wrapper → inference pipeline → hierarchical decision → frontend result
```

## Demo Image Result

Input image:

```text
ml/data/local_unknown/images/local_000001.jpg
```

Output:

| Field            | Value                |
| ---------------- | -------------------- |
| pred_label       | plastic              |
| confidence       | 0.962933             |
| top_coarse_label | recyclable           |
| decision_type    | fine_label           |
| final_label      | plastic              |
| reason           | fine_confidence_high |

## Key Technical Work Completed

| Area                  | Completed Work                                                         |
| --------------------- | ---------------------------------------------------------------------- |
| Dataset               | manifest system, TrashNet split, local unknown workflow                |
| Model                 | baseline MobileNetV3-style classifier                                  |
| Evaluation            | closed-set metrics, reject baselines, open-set score baselines         |
| Safety                | confidence reject option, hierarchical fallback, manual-review routing |
| Local unknown testing | tested reject behaviour on local unknown images                        |
| Policy tuning         | selected safer hierarchical policy                                     |
| Active learning       | selected 20 candidates for human review                                |
| Human labelling       | generated labelling sheet and processing workflow                      |
| Inference             | single-image and batch inference pipelines                             |
| Backend               | FastAPI `/health` and `/api/inference/predict` endpoints               |
| Frontend              | simple browser demo page                                               |
| Documentation         | run guide and demo checklist                                           |

## Important Results

| Experiment                 | Main Result                                           |
| -------------------------- | ----------------------------------------------------- |
| Closed-set baseline        | test accuracy 0.692708                                |
| Confidence reject baseline | local unknown rejection rate 0.350000                 |
| First hierarchical policy  | high known coverage but too permissive for unknowns   |
| Safe hierarchical policy   | local unknown manual-review rate improved to 0.375000 |
| Active learning            | 20 local candidates selected for human labelling      |
| Frontend/backend demo      | working successfully                                  |

## What Makes the Project Different

This project should not be presented as only a CNN waste classifier.

The stronger contribution is:

```text
hierarchical open-set waste classification with reject option and local active learning
```

The system is designed to avoid unsafe forced predictions by returning coarse fallback labels or manual-review decisions when needed.

## Current Limitations

| Limitation                             | Explanation                                           |
| -------------------------------------- | ----------------------------------------------------- |
| Known model labels are limited         | current model has five trained fine labels            |
| Local unknown false acceptance remains | tuned policy improves safety but is not perfect       |
| Human review not completed yet         | active learning sheet is ready but pending annotation |
| Frontend uses image path               | upload support can be added later                     |
| Baseline model can be improved         | pretrained training can be future work                |

## Suggested Demo Script

1. Explain the problem: normal waste classifiers force a known class.
2. Explain OpenWaste-HR: fine label, coarse fallback, or manual review.
3. Start the backend.
4. Open the frontend.
5. Run the demo image.
6. Show final label: plastic.
7. Show confidence and class probabilities.
8. Explain how unknown/manual-review and active learning support real-world deployment.

## Next Recommended Step

The next useful step is to prepare a thesis-ready implementation summary and architecture diagram showing the full OpenWaste-HR workflow.
