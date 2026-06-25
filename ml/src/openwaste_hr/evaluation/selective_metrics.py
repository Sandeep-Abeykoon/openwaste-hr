from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def apply_confidence_threshold(
    predictions_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Apply confidence-threshold rejection to a prediction DataFrame.

    Required columns:
    - true_label_id
    - pred_label_id
    - true_label
    - pred_label
    - max_softmax_confidence
    - correct
    """
    required_columns = {
        "true_label_id",
        "pred_label_id",
        "true_label",
        "pred_label",
        "max_softmax_confidence",
        "correct",
    }

    missing_columns = required_columns - set(predictions_df.columns)
    if missing_columns:
        raise ValueError(
            f"Predictions DataFrame is missing required columns: {sorted(missing_columns)}"
        )

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    result = predictions_df.copy()

    result["confidence_threshold"] = float(threshold)
    result["is_accepted"] = (
        result["max_softmax_confidence"].astype(float) >= float(threshold)
    )

    result["decision"] = result["is_accepted"].map(
        {
            True: "accept_fine_label",
            False: "manual_review",
        }
    )

    result["final_label"] = result.apply(
        lambda row: row["pred_label"] if row["is_accepted"] else "manual_review",
        axis=1,
    )

    result["reject_reason"] = result["is_accepted"].map(
        {
            True: "",
            False: "confidence_below_threshold",
        }
    )

    return result


def compute_selective_metrics(
    thresholded_df: pd.DataFrame,
    label_names: list[str],
) -> dict[str, Any]:
    """
    Compute selective classification metrics after rejection.

    Coverage = accepted samples / all samples.
    Selective accuracy = accuracy among accepted samples only.
    Selective error rate = 1 - selective accuracy.
    """
    required_columns = {
        "true_label_id",
        "pred_label_id",
        "correct",
        "is_accepted",
    }

    missing_columns = required_columns - set(thresholded_df.columns)
    if missing_columns:
        raise ValueError(
            f"Thresholded DataFrame is missing required columns: {sorted(missing_columns)}"
        )

    total_samples = len(thresholded_df)
    if total_samples == 0:
        raise ValueError("Cannot compute metrics for an empty DataFrame.")

    accepted_df = thresholded_df[thresholded_df["is_accepted"] == True].copy()  # noqa: E712
    rejected_df = thresholded_df[thresholded_df["is_accepted"] == False].copy()  # noqa: E712

    accepted_count = len(accepted_df)
    rejected_count = len(rejected_df)

    coverage = accepted_count / total_samples
    rejection_rate = rejected_count / total_samples

    forced_accuracy = accuracy_score(
        thresholded_df["true_label_id"].astype(int),
        thresholded_df["pred_label_id"].astype(int),
    )

    if accepted_count == 0:
        selective_accuracy = 0.0
        selective_error_rate = 1.0
        selective_macro_f1 = 0.0
        selective_weighted_f1 = 0.0
    else:
        y_true_accepted = accepted_df["true_label_id"].astype(int).tolist()
        y_pred_accepted = accepted_df["pred_label_id"].astype(int).tolist()
        labels = list(range(len(label_names)))

        selective_accuracy = accuracy_score(
            y_true_accepted,
            y_pred_accepted,
        )
        selective_error_rate = 1.0 - selective_accuracy
        selective_macro_f1 = f1_score(
            y_true_accepted,
            y_pred_accepted,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        selective_weighted_f1 = f1_score(
            y_true_accepted,
            y_pred_accepted,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

    return {
        "total_samples": int(total_samples),
        "accepted_count": int(accepted_count),
        "rejected_count": int(rejected_count),
        "coverage": round(float(coverage), 6),
        "rejection_rate": round(float(rejection_rate), 6),
        "forced_accuracy": round(float(forced_accuracy), 6),
        "selective_accuracy": round(float(selective_accuracy), 6),
        "selective_error_rate": round(float(selective_error_rate), 6),
        "selective_macro_f1": round(float(selective_macro_f1), 6),
        "selective_weighted_f1": round(float(selective_weighted_f1), 6),
    }


def sweep_confidence_thresholds(
    predictions_df: pd.DataFrame,
    label_names: list[str],
    thresholds: list[float],
) -> pd.DataFrame:
    """
    Evaluate many confidence thresholds.
    """
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        thresholded_df = apply_confidence_threshold(
            predictions_df=predictions_df,
            threshold=float(threshold),
        )
        metrics = compute_selective_metrics(
            thresholded_df=thresholded_df,
            label_names=label_names,
        )

        rows.append(
            {
                "threshold": round(float(threshold), 6),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def select_threshold_from_sweep(
    sweep_df: pd.DataFrame,
    min_coverage: float,
    selection_metric: str = "selective_macro_f1",
) -> dict[str, Any]:
    """
    Select the best threshold using validation results.

    The selected threshold must satisfy minimum coverage.
    Among valid thresholds, choose the highest selection metric.
    """
    required_columns = {
        "threshold",
        "coverage",
        "selective_accuracy",
        "selective_macro_f1",
    }

    missing_columns = required_columns - set(sweep_df.columns)
    if missing_columns:
        raise ValueError(f"Sweep DataFrame missing columns: {sorted(missing_columns)}")

    if selection_metric not in sweep_df.columns:
        raise ValueError(f"Selection metric not found in sweep DataFrame: {selection_metric}")

    if min_coverage <= 0.0 or min_coverage > 1.0:
        raise ValueError("min_coverage must be greater than 0.0 and at most 1.0.")

    valid_df = sweep_df[sweep_df["coverage"] >= float(min_coverage)].copy()

    if valid_df.empty:
        raise ValueError(
            "No threshold satisfies the minimum coverage requirement. "
            f"min_coverage={min_coverage}"
        )

    selected_row = (
        valid_df.sort_values(
            by=[selection_metric, "selective_accuracy", "coverage", "threshold"],
            ascending=[False, False, False, True],
        )
        .iloc[0]
        .to_dict()
    )

    selected_row["selection_metric"] = selection_metric
    selected_row["min_coverage"] = float(min_coverage)

    return selected_row