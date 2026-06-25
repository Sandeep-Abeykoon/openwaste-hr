from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_max_logit_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute Maximum Logit Score.

    Higher max logit means stronger evidence for a known class.
    """
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor with shape [batch_size, num_classes].")

    return torch.max(logits, dim=1).values


def compute_energy_score(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute energy score from logits.

    energy = -T * logsumexp(logits / T)

    Lower energy means stronger in-distribution evidence.
    """
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor with shape [batch_size, num_classes].")

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")

    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def build_thresholds_from_scores(
    scores: pd.Series,
    threshold_count: int = 101,
) -> list[float]:
    """
    Build evenly spaced thresholds between min and max score values.
    """
    if threshold_count < 2:
        raise ValueError("threshold_count must be at least 2.")

    clean_scores = scores.astype(float).dropna()

    if clean_scores.empty:
        raise ValueError("Cannot build thresholds from an empty score series.")

    min_score = float(clean_scores.min())
    max_score = float(clean_scores.max())

    if min_score == max_score:
        return [round(min_score, 6)]

    step = (max_score - min_score) / (threshold_count - 1)

    return [
        round(min_score + index * step, 6)
        for index in range(threshold_count)
    ]


def apply_score_threshold(
    predictions_df: pd.DataFrame,
    score_column: str,
    threshold: float,
    accept_direction: str,
) -> pd.DataFrame:
    """
    Apply threshold rejection using a generic score column.

    accept_direction:
    - greater_equal: accept when score >= threshold
    - less_equal: accept when score <= threshold
    """
    required_columns = {
        "true_label_id",
        "pred_label_id",
        "true_label",
        "pred_label",
        "correct",
        score_column,
    }

    missing_columns = required_columns - set(predictions_df.columns)
    if missing_columns:
        raise ValueError(
            f"Predictions DataFrame is missing required columns: {sorted(missing_columns)}"
        )

    if accept_direction not in {"greater_equal", "less_equal"}:
        raise ValueError(
            "accept_direction must be either 'greater_equal' or 'less_equal'."
        )

    result = predictions_df.copy()
    result["score_column"] = score_column
    result["score_threshold"] = float(threshold)
    result["accept_direction"] = accept_direction

    if accept_direction == "greater_equal":
        result["is_accepted"] = result[score_column].astype(float) >= float(threshold)
        reject_reason = f"{score_column}_below_threshold"
    else:
        result["is_accepted"] = result[score_column].astype(float) <= float(threshold)
        reject_reason = f"{score_column}_above_threshold"

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
            False: reject_reason,
        }
    )

    return result


def compute_score_selective_metrics(
    thresholded_df: pd.DataFrame,
    label_names: list[str],
) -> dict[str, Any]:
    """
    Compute selective classification metrics after score-based rejection.
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


def sweep_score_thresholds(
    predictions_df: pd.DataFrame,
    label_names: list[str],
    score_name: str,
    score_column: str,
    thresholds: list[float],
    accept_direction: str,
) -> pd.DataFrame:
    """
    Evaluate many score thresholds.
    """
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        thresholded_df = apply_score_threshold(
            predictions_df=predictions_df,
            score_column=score_column,
            threshold=float(threshold),
            accept_direction=accept_direction,
        )

        metrics = compute_score_selective_metrics(
            thresholded_df=thresholded_df,
            label_names=label_names,
        )

        rows.append(
            {
                "score_name": score_name,
                "score_column": score_column,
                "threshold": round(float(threshold), 6),
                "accept_direction": accept_direction,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def select_score_threshold_from_sweep(
    sweep_df: pd.DataFrame,
    min_coverage: float,
    selection_metric: str,
) -> dict[str, Any]:
    """
    Select the best score threshold using validation results only.
    """
    required_columns = {
        "score_name",
        "threshold",
        "coverage",
        "selective_accuracy",
        "selective_macro_f1",
        "accept_direction",
    }

    missing_columns = required_columns - set(sweep_df.columns)
    if missing_columns:
        raise ValueError(f"Sweep DataFrame missing columns: {sorted(missing_columns)}")

    if selection_metric not in sweep_df.columns:
        raise ValueError(f"Selection metric not found: {selection_metric}")

    if min_coverage <= 0.0 or min_coverage > 1.0:
        raise ValueError("min_coverage must be greater than 0.0 and at most 1.0.")

    valid_df = sweep_df[sweep_df["coverage"] >= float(min_coverage)].copy()

    if valid_df.empty:
        raise ValueError(
            "No score threshold satisfies the minimum coverage requirement. "
            f"min_coverage={min_coverage}"
        )

    selected_row = (
        valid_df.sort_values(
            by=[selection_metric, "selective_accuracy", "coverage"],
            ascending=[False, False, False],
        )
        .iloc[0]
        .to_dict()
    )

    selected_row["selection_metric"] = selection_metric
    selected_row["min_coverage"] = float(min_coverage)

    return selected_row