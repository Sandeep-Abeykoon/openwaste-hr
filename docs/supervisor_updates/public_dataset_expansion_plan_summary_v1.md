# Public Dataset Expansion Plan Summary v1

## Purpose

This stage prepares the next research phase for OpenWaste-HR by planning public dataset expansion.

## Why This Is Needed

The current v1 pipeline is complete, but it was trained mainly using the TrashNet-style dataset.

This is useful for the first working system, but it is not enough for the strongest final research version.

## Current Situation

| System                              | Status       |
| ----------------------------------- | ------------ |
| TrashNet scratch baseline           | complete     |
| TrashNet pretrained baseline        | complete     |
| safe pretrained hierarchical policy | complete     |
| backend/frontend prototype          | complete     |
| active learning v2 seed             | complete     |
| public dataset expansion            | planned next |

## Recommended Next Dataset

The recommended next dataset is:

```text id="sww6ju"
RealWaste
```

Reason:

RealWaste is closer to the current image-classification workflow and should be easier to integrate before attempting more complex annotation-based datasets.

## Later Dataset Candidate

```text id="xxfk3b"
TACO
```

TACO is useful later for in-the-wild waste evaluation and hierarchical/open-set analysis, but it may require more preprocessing because it is more annotation-oriented.

## Expansion Rule

Do not force unclear labels or outside-taxonomy objects into current known classes.

Each new dataset label should be mapped as one of:

* known_train_candidate
* known_eval_candidate
* unknown_eval_candidate
* future_class_candidate
* exclude_or_review

## Next Implementation Step

The next implementation step should create:

1. RealWaste intake config
2. RealWaste source label mapping template
3. RealWaste manifest builder
4. RealWaste manifest validation test
5. RealWaste inspection report

After that, the model can be retrained and compared against the current best pretrained safe policy.
