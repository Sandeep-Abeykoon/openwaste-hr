# Reading Log

## Priority References

### 1. TACO: Trash Annotations in Context for Litter Detection
Use for: in-the-wild waste/litter detection challenge, annotation limitations, dataset realism.

### 2. Open-Set Recognition: A Good Closed-Set Classifier is All You Need?
Use for: open-set recognition, maximum logit baseline, unknown detection justification.

### 3. PyTorch Documentation
Use for: model training implementation.

### 4. FastAPI Documentation
Use for: backend API implementation.

## How references will be used

Each implementation feature must connect to a research reason:

| Implementation Feature | Research Reason |
|---|---|
| Fine-label classifier | Standard baseline for known classes |
| Coarse-label fallback | Handles fine-class uncertainty more safely |
| Unknown/reject option | Avoids confident wrong predictions |
| Local correction storage | Supports active learning and local adaptation |
| Evaluation with unknown classes | Tests open-world reliability |