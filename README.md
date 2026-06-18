# OpenWaste-HR

OpenWaste-HR is a final-year software engineering research project focused on hierarchical open-set waste classification.

## Project Title

OpenWaste-HR: Hierarchical Open-Set Waste Classification with Reject Option and Local Active Learning

## Core Research Problem

Most waste classification systems behave as closed-set classifiers, forcing every input image into a known class. Real waste streams are open-world, noisy, shifted, and may contain unknown, damaged, mixed, or ambiguous items.

## Proposed Contribution

This project aims to build a waste classification system that can:

1. Predict a fine-grained waste class when confidence is high.
2. Fall back to a coarse category when fine-level prediction is uncertain.
3. Reject uncertain or unknown items for manual review.
4. Store uncertain samples for local active learning and later model improvement.

## Main Components

- Hierarchical coarse/fine waste taxonomy
- Closed-set baseline classifier
- Confidence-threshold and open-set reject baseline
- Proposed hierarchical open-set classifier
- Local correction and active-learning loop
- API and user interface for demonstration

## Planned Stack

- Python
- PyTorch
- FastAPI
- React
- ONNX Runtime
- Scikit-learn
- OpenCV