from __future__ import annotations

from typing import Any

import pandas as pd


def get_probability_column_name(label_name: str) -> str:
    """
    Return the probability column name used by prediction CSV files.
    """
    return f"prob_{label_name}"


def validate_fine_to_coarse_mapping(
    fine_to_coarse: dict[str, str],
) -> bool:
    """
    Validate fine-to-coarse mapping.
    """
    if not fine_to_coarse:
        raise ValueError("fine_to_coarse mapping cannot be empty.")

    for fine_label, coarse_label in fine_to_coarse.items():
        if not str(fine_label).strip():
            raise ValueError("fine_to_coarse contains an empty fine label.")

        if not str(coarse_label).strip():
            raise ValueError("fine_to_coarse contains an empty coarse label.")

    return True


def validate_prediction_columns(
    predictions_df: pd.DataFrame,
    fine_to_coarse: dict[str, str],
) -> bool:
    """
    Validate required columns for hierarchical decision policy.
    """
    validate_fine_to_coarse_mapping(fine_to_coarse)

    required_columns = {
        "sample_id",
        "image_path",
        "pred_label",
        "max_softmax_confidence",
    }

    missing_columns = required_columns - set(predictions_df.columns)

    if missing_columns:
        raise ValueError(
            f"Predictions DataFrame is missing required columns: {sorted(missing_columns)}"
        )

    missing_probability_columns = []

    for fine_label in fine_to_coarse:
        probability_column = get_probability_column_name(fine_label)

        if probability_column not in predictions_df.columns:
            missing_probability_columns.append(probability_column)

    if missing_probability_columns:
        raise ValueError(
            "Predictions DataFrame is missing probability columns: "
            f"{missing_probability_columns}"
        )

    return True


def compute_coarse_probabilities_for_row(
    row: pd.Series,
    fine_to_coarse: dict[str, str],
) -> dict[str, float]:
    """
    Aggregate fine-label probabilities into coarse-label probabilities.
    """
    coarse_probabilities: dict[str, float] = {}

    for fine_label, coarse_label in fine_to_coarse.items():
        probability_column = get_probability_column_name(fine_label)
        probability_value = float(row[probability_column])

        coarse_probabilities[coarse_label] = (
            coarse_probabilities.get(coarse_label, 0.0) + probability_value
        )

    return coarse_probabilities


def add_hierarchical_scores(
    predictions_df: pd.DataFrame,
    fine_to_coarse: dict[str, str],
) -> pd.DataFrame:
    """
    Add top coarse label, coarse confidence, and coarse margin.
    """
    validate_prediction_columns(
        predictions_df=predictions_df,
        fine_to_coarse=fine_to_coarse,
    )

    result = predictions_df.copy()

    top_coarse_labels: list[str] = []
    top_coarse_confidences: list[float] = []
    second_coarse_confidences: list[float] = []
    coarse_margins: list[float] = []

    for _, row in result.iterrows():
        coarse_probabilities = compute_coarse_probabilities_for_row(
            row=row,
            fine_to_coarse=fine_to_coarse,
        )

        sorted_coarse = sorted(
            coarse_probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_label, top_confidence = sorted_coarse[0]

        if len(sorted_coarse) > 1:
            second_confidence = sorted_coarse[1][1]
        else:
            second_confidence = 0.0

        top_coarse_labels.append(top_label)
        top_coarse_confidences.append(round(float(top_confidence), 6))
        second_coarse_confidences.append(round(float(second_confidence), 6))
        coarse_margins.append(round(float(top_confidence - second_confidence), 6))

    result["top_coarse_label"] = top_coarse_labels
    result["top_coarse_confidence"] = top_coarse_confidences
    result["second_coarse_confidence"] = second_coarse_confidences
    result["coarse_margin"] = coarse_margins

    return result


def apply_hierarchical_policy(
    predictions_df: pd.DataFrame,
    fine_to_coarse: dict[str, str],
    fine_confidence_threshold: float,
    coarse_confidence_threshold: float,
    coarse_margin_threshold: float,
    minimum_confidence_for_coarse: float,
) -> pd.DataFrame:
    """
    Apply hierarchical fine/coarse/manual-review decision policy.

    Decision order:
    1. Fine label when fine confidence is high.
    2. Coarse label when fine confidence is lower but coarse evidence is stable.
    3. Manual review otherwise.
    """
    if not 0.0 <= fine_confidence_threshold <= 1.0:
        raise ValueError("fine_confidence_threshold must be between 0 and 1.")

    if not 0.0 <= coarse_confidence_threshold <= 1.0:
        raise ValueError("coarse_confidence_threshold must be between 0 and 1.")

    if not 0.0 <= coarse_margin_threshold <= 1.0:
        raise ValueError("coarse_margin_threshold must be between 0 and 1.")

    if not 0.0 <= minimum_confidence_for_coarse <= 1.0:
        raise ValueError("minimum_confidence_for_coarse must be between 0 and 1.")

    result = add_hierarchical_scores(
        predictions_df=predictions_df,
        fine_to_coarse=fine_to_coarse,
    )

    decision_types: list[str] = []
    final_labels: list[str] = []
    final_confidences: list[float] = []
    decision_reasons: list[str] = []

    for _, row in result.iterrows():
        fine_confidence = float(row["max_softmax_confidence"])
        top_coarse_confidence = float(row["top_coarse_confidence"])
        coarse_margin = float(row["coarse_margin"])

        if fine_confidence >= fine_confidence_threshold:
            decision_types.append("fine_label")
            final_labels.append(str(row["pred_label"]))
            final_confidences.append(round(float(fine_confidence), 6))
            decision_reasons.append("fine_confidence_high")

        elif (
            fine_confidence >= minimum_confidence_for_coarse
            and top_coarse_confidence >= coarse_confidence_threshold
            and coarse_margin >= coarse_margin_threshold
        ):
            decision_types.append("coarse_label")
            final_labels.append(str(row["top_coarse_label"]))
            final_confidences.append(round(float(top_coarse_confidence), 6))
            decision_reasons.append("coarse_fallback_stable")

        else:
            decision_types.append("manual_review")
            final_labels.append("manual_review")
            final_confidences.append(round(float(fine_confidence), 6))
            decision_reasons.append("unsafe_or_uncertain")

    result["hierarchical_decision_type"] = decision_types
    result["hierarchical_final_label"] = final_labels
    result["hierarchical_final_confidence"] = final_confidences
    result["hierarchical_decision_reason"] = decision_reasons

    return result


def compute_known_hierarchical_metrics(
    decisions_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute metrics for known-class test images.

    Fine output is correct when final label equals true fine label.
    Coarse output is correct when final label equals true coarse label.
    Manual review is treated as abstention.
    """
    required_columns = {
        "true_label",
        "coarse_label",
        "hierarchical_decision_type",
        "hierarchical_final_label",
    }

    missing_columns = required_columns - set(decisions_df.columns)

    if missing_columns:
        raise ValueError(
            f"Known decisions DataFrame is missing columns: {sorted(missing_columns)}"
        )

    total_samples = len(decisions_df)

    if total_samples == 0:
        raise ValueError("Cannot compute known metrics on an empty DataFrame.")

    fine_df = decisions_df[
        decisions_df["hierarchical_decision_type"] == "fine_label"
    ].copy()

    coarse_df = decisions_df[
        decisions_df["hierarchical_decision_type"] == "coarse_label"
    ].copy()

    manual_df = decisions_df[
        decisions_df["hierarchical_decision_type"] == "manual_review"
    ].copy()

    fine_decision_count = len(fine_df)
    coarse_fallback_count = len(coarse_df)
    manual_review_count = len(manual_df)
    accepted_count = fine_decision_count + coarse_fallback_count

    fine_correct_count = int(
        (
            fine_df["hierarchical_final_label"].astype(str)
            == fine_df["true_label"].astype(str)
        ).sum()
    )

    coarse_correct_count = int(
        (
            coarse_df["hierarchical_final_label"].astype(str)
            == coarse_df["coarse_label"].astype(str)
        ).sum()
    )

    hierarchical_success_count = fine_correct_count + coarse_correct_count

    fine_accuracy = (
        fine_correct_count / fine_decision_count
        if fine_decision_count > 0
        else 0.0
    )

    coarse_accuracy = (
        coarse_correct_count / coarse_fallback_count
        if coarse_fallback_count > 0
        else 0.0
    )

    success_rate_over_all = hierarchical_success_count / total_samples

    success_rate_over_accepted = (
        hierarchical_success_count / accepted_count
        if accepted_count > 0
        else 0.0
    )

    return {
        "known_total_samples": int(total_samples),
        "fine_decision_count": int(fine_decision_count),
        "coarse_fallback_count": int(coarse_fallback_count),
        "manual_review_count": int(manual_review_count),
        "known_decision_coverage": round(float(accepted_count / total_samples), 6),
        "known_manual_review_rate": round(float(manual_review_count / total_samples), 6),
        "fine_correct_count": int(fine_correct_count),
        "coarse_correct_count": int(coarse_correct_count),
        "hierarchical_success_count": int(hierarchical_success_count),
        "fine_accuracy_on_fine_decisions": round(float(fine_accuracy), 6),
        "coarse_accuracy_on_coarse_decisions": round(float(coarse_accuracy), 6),
        "hierarchical_success_rate_over_all": round(float(success_rate_over_all), 6),
        "hierarchical_success_rate_over_accepted": round(
            float(success_rate_over_accepted),
            6,
        ),
    }


def compute_unknown_hierarchical_metrics(
    decisions_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute metrics for local unknown images.

    Manual review is desired.
    Fine or coarse output is treated as accepting an unknown image.
    """
    required_columns = {
        "hierarchical_decision_type",
        "hierarchical_final_label",
    }

    missing_columns = required_columns - set(decisions_df.columns)

    if missing_columns:
        raise ValueError(
            f"Unknown decisions DataFrame is missing columns: {sorted(missing_columns)}"
        )

    total_samples = len(decisions_df)

    if total_samples == 0:
        raise ValueError("Cannot compute unknown metrics on an empty DataFrame.")

    fine_accept_count = int(
        (decisions_df["hierarchical_decision_type"] == "fine_label").sum()
    )

    coarse_accept_count = int(
        (decisions_df["hierarchical_decision_type"] == "coarse_label").sum()
    )

    manual_review_count = int(
        (decisions_df["hierarchical_decision_type"] == "manual_review").sum()
    )

    accepted_unknown_count = fine_accept_count + coarse_accept_count

    return {
        "unknown_total_samples": int(total_samples),
        "unknown_manual_review_count": int(manual_review_count),
        "unknown_fine_accept_count": int(fine_accept_count),
        "unknown_coarse_accept_count": int(coarse_accept_count),
        "unknown_accepted_count": int(accepted_unknown_count),
        "unknown_manual_review_rate": round(float(manual_review_count / total_samples), 6),
        "unknown_acceptance_rate": round(float(accepted_unknown_count / total_samples), 6),
    }


def build_hierarchical_decision_distribution(
    known_decisions_df: pd.DataFrame,
    unknown_decisions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build decision-type counts for known and local unknown datasets.
    """
    rows: list[dict[str, Any]] = []

    for dataset_name, dataframe in [
        ("known_test", known_decisions_df),
        ("local_unknown", unknown_decisions_df),
    ]:
        total = len(dataframe)

        counts = (
            dataframe["hierarchical_decision_type"]
            .value_counts()
            .to_dict()
        )

        for decision_type in ["fine_label", "coarse_label", "manual_review"]:
            count = int(counts.get(decision_type, 0))

            rows.append(
                {
                    "dataset": dataset_name,
                    "decision_type": decision_type,
                    "count": count,
                    "percentage": round(float(count / total * 100), 2)
                    if total > 0
                    else 0.0,
                }
            )

    return pd.DataFrame(rows)