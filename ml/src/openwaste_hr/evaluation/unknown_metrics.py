from __future__ import annotations

from typing import Any

import pandas as pd


def apply_unknown_decision_rule(
    predictions_df: pd.DataFrame,
    method_name: str,
    score_column: str,
    threshold: float,
    accept_direction: str,
) -> pd.DataFrame:
    """
    Apply a reject/accept rule to unknown samples.

    For unknown evaluation:
    - rejected/manual_review is desirable
    - accepted as known label is a false acceptance
    """
    required_columns = {
        "sample_id",
        "image_path",
        "pred_label",
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
    result["method_name"] = method_name
    result["score_column"] = score_column
    result["threshold"] = float(threshold)
    result["accept_direction"] = accept_direction

    if accept_direction == "greater_equal":
        result["is_accepted_as_known"] = (
            result[score_column].astype(float) >= float(threshold)
        )
    else:
        result["is_accepted_as_known"] = (
            result[score_column].astype(float) <= float(threshold)
        )

    result["is_rejected_as_unknown"] = ~result["is_accepted_as_known"]

    result["final_decision"] = result["is_accepted_as_known"].map(
        {
            True: "accepted_as_known",
            False: "manual_review",
        }
    )

    result["final_label"] = result.apply(
        lambda row: row["pred_label"] if row["is_accepted_as_known"] else "manual_review",
        axis=1,
    )

    result["unknown_eval_outcome"] = result["is_accepted_as_known"].map(
        {
            True: "false_accept_unknown_as_known",
            False: "correct_reject_unknown",
        }
    )

    return result


def compute_unknown_rejection_metrics(
    decisions_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute metrics for unknown/manual-review evaluation.
    """
    required_columns = {
        "method_name",
        "is_accepted_as_known",
        "is_rejected_as_unknown",
    }

    missing_columns = required_columns - set(decisions_df.columns)
    if missing_columns:
        raise ValueError(
            f"Decisions DataFrame is missing required columns: {sorted(missing_columns)}"
        )

    total_samples = len(decisions_df)

    if total_samples == 0:
        raise ValueError("Cannot compute unknown metrics for an empty DataFrame.")

    accepted_count = int(decisions_df["is_accepted_as_known"].sum())
    rejected_count = int(decisions_df["is_rejected_as_unknown"].sum())

    unknown_rejection_rate = rejected_count / total_samples
    unknown_false_acceptance_rate = accepted_count / total_samples

    return {
        "total_unknown_samples": int(total_samples),
        "rejected_unknown_count": rejected_count,
        "accepted_unknown_as_known_count": accepted_count,
        "unknown_rejection_rate": round(float(unknown_rejection_rate), 6),
        "unknown_false_acceptance_rate": round(float(unknown_false_acceptance_rate), 6),
    }


def build_accepted_label_distribution(
    decisions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build predicted-label distribution for unknown samples accepted as known.
    """
    accepted_df = decisions_df[
        decisions_df["is_accepted_as_known"] == True  # noqa: E712
    ].copy()

    if accepted_df.empty:
        return pd.DataFrame(
            columns=[
                "method_name",
                "pred_label",
                "accepted_count",
                "accepted_percentage_within_method",
            ]
        )

    grouped = (
        accepted_df.groupby(["method_name", "pred_label"])
        .size()
        .reset_index(name="accepted_count")
    )

    total_by_method = (
        accepted_df.groupby("method_name")
        .size()
        .reset_index(name="method_accepted_total")
    )

    merged = grouped.merge(total_by_method, on="method_name", how="left")

    merged["accepted_percentage_within_method"] = (
        merged["accepted_count"] / merged["method_accepted_total"] * 100
    ).round(2)

    return merged.drop(columns=["method_accepted_total"])