from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_summary_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> dict[str, float]:
    """
    Compute main classification metrics.
    """
    labels = list(range(len(label_names)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def build_confusion_matrix_dataframe(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> pd.DataFrame:
    """
    Build a labelled confusion matrix DataFrame.
    """
    labels = list(range(len(label_names)))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in label_names],
        columns=[f"pred_{label}" for label in label_names],
    )


def build_classification_report_dataframe(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> pd.DataFrame:
    """
    Build a classification report DataFrame.
    """
    labels = list(range(len(label_names)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    return (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )


def build_prediction_dataframe(
    metadata_rows: list[dict[str, Any]],
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    probability_rows: list[list[float]],
    label_names: list[str],
) -> pd.DataFrame:
    """
    Build per-image prediction table.

    This file is useful later for confidence-threshold and reject-option analysis.
    """
    rows: list[dict[str, Any]] = []

    for index, metadata in enumerate(metadata_rows):
        true_id = int(y_true[index])
        pred_id = int(y_pred[index])

        row = {
            **metadata,
            "true_label_id": true_id,
            "true_label": label_names[true_id],
            "pred_label_id": pred_id,
            "pred_label": label_names[pred_id],
            "correct": true_id == pred_id,
            "max_softmax_confidence": round(float(confidences[index]), 6),
        }

        for label_index, label_name in enumerate(label_names):
            row[f"prob_{label_name}"] = round(
                float(probability_rows[index][label_index]),
                6,
            )

        rows.append(row)

    return pd.DataFrame(rows)