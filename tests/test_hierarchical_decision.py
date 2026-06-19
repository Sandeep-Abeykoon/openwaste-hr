import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.evaluation.hierarchical_decision import (  # noqa: E402
    add_hierarchical_scores,
    apply_hierarchical_policy,
    build_hierarchical_decision_distribution,
    compute_known_hierarchical_metrics,
    compute_unknown_hierarchical_metrics,
)


FINE_TO_COARSE = {
    "paper_cardboard": "recyclable",
    "plastic": "recyclable",
    "glass": "recyclable",
    "metal": "recyclable",
    "residual": "residual",
}


def create_known_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "known_001",
                "image_path": "known_001.jpg",
                "true_label": "plastic",
                "coarse_label": "recyclable",
                "pred_label": "plastic",
                "max_softmax_confidence": 0.90,
                "prob_paper_cardboard": 0.02,
                "prob_plastic": 0.90,
                "prob_glass": 0.03,
                "prob_metal": 0.03,
                "prob_residual": 0.02,
            },
            {
                "sample_id": "known_002",
                "image_path": "known_002.jpg",
                "true_label": "glass",
                "coarse_label": "recyclable",
                "pred_label": "plastic",
                "max_softmax_confidence": 0.40,
                "prob_paper_cardboard": 0.20,
                "prob_plastic": 0.40,
                "prob_glass": 0.25,
                "prob_metal": 0.05,
                "prob_residual": 0.10,
            },
            {
                "sample_id": "known_003",
                "image_path": "known_003.jpg",
                "true_label": "residual",
                "coarse_label": "residual",
                "pred_label": "paper_cardboard",
                "max_softmax_confidence": 0.30,
                "prob_paper_cardboard": 0.30,
                "prob_plastic": 0.20,
                "prob_glass": 0.15,
                "prob_metal": 0.10,
                "prob_residual": 0.25,
            },
        ]
    )


def create_unknown_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "local_001",
                "image_path": "local_001.jpg",
                "pred_label": "plastic",
                "max_softmax_confidence": 0.91,
                "prob_paper_cardboard": 0.02,
                "prob_plastic": 0.91,
                "prob_glass": 0.03,
                "prob_metal": 0.02,
                "prob_residual": 0.02,
            },
            {
                "sample_id": "local_002",
                "image_path": "local_002.jpg",
                "pred_label": "metal",
                "max_softmax_confidence": 0.25,
                "prob_paper_cardboard": 0.20,
                "prob_plastic": 0.25,
                "prob_glass": 0.20,
                "prob_metal": 0.15,
                "prob_residual": 0.20,
            },
        ]
    )


def test_add_hierarchical_scores():
    prediction_df = create_known_prediction_df()

    scored_df = add_hierarchical_scores(
        predictions_df=prediction_df,
        fine_to_coarse=FINE_TO_COARSE,
    )

    assert scored_df.loc[0, "top_coarse_label"] == "recyclable"
    assert scored_df.loc[0, "top_coarse_confidence"] == 0.98
    assert "coarse_margin" in scored_df.columns


def test_apply_hierarchical_policy_outputs_fine_coarse_and_manual_review():
    prediction_df = create_known_prediction_df()

    decisions_df = apply_hierarchical_policy(
        predictions_df=prediction_df,
        fine_to_coarse=FINE_TO_COARSE,
        fine_confidence_threshold=0.64,
        coarse_confidence_threshold=0.80,
        coarse_margin_threshold=0.15,
        minimum_confidence_for_coarse=0.35,
    )

    assert decisions_df.loc[0, "hierarchical_decision_type"] == "fine_label"
    assert decisions_df.loc[0, "hierarchical_final_label"] == "plastic"

    assert decisions_df.loc[1, "hierarchical_decision_type"] == "coarse_label"
    assert decisions_df.loc[1, "hierarchical_final_label"] == "recyclable"

    assert decisions_df.loc[2, "hierarchical_decision_type"] == "manual_review"
    assert decisions_df.loc[2, "hierarchical_final_label"] == "manual_review"


def test_compute_known_hierarchical_metrics():
    prediction_df = create_known_prediction_df()

    decisions_df = apply_hierarchical_policy(
        predictions_df=prediction_df,
        fine_to_coarse=FINE_TO_COARSE,
        fine_confidence_threshold=0.64,
        coarse_confidence_threshold=0.80,
        coarse_margin_threshold=0.15,
        minimum_confidence_for_coarse=0.35,
    )

    metrics = compute_known_hierarchical_metrics(decisions_df)

    assert metrics["known_total_samples"] == 3
    assert metrics["fine_decision_count"] == 1
    assert metrics["coarse_fallback_count"] == 1
    assert metrics["manual_review_count"] == 1
    assert metrics["hierarchical_success_count"] == 2


def test_compute_unknown_hierarchical_metrics():
    prediction_df = create_unknown_prediction_df()

    decisions_df = apply_hierarchical_policy(
        predictions_df=prediction_df,
        fine_to_coarse=FINE_TO_COARSE,
        fine_confidence_threshold=0.64,
        coarse_confidence_threshold=0.80,
        coarse_margin_threshold=0.15,
        minimum_confidence_for_coarse=0.35,
    )

    metrics = compute_unknown_hierarchical_metrics(decisions_df)

    assert metrics["unknown_total_samples"] == 2
    assert metrics["unknown_fine_accept_count"] == 1
    assert metrics["unknown_manual_review_count"] == 1
    assert metrics["unknown_acceptance_rate"] == 0.5


def test_build_hierarchical_decision_distribution():
    known_df = create_known_prediction_df()
    unknown_df = create_unknown_prediction_df()

    known_decisions = apply_hierarchical_policy(
        predictions_df=known_df,
        fine_to_coarse=FINE_TO_COARSE,
        fine_confidence_threshold=0.64,
        coarse_confidence_threshold=0.80,
        coarse_margin_threshold=0.15,
        minimum_confidence_for_coarse=0.35,
    )

    unknown_decisions = apply_hierarchical_policy(
        predictions_df=unknown_df,
        fine_to_coarse=FINE_TO_COARSE,
        fine_confidence_threshold=0.64,
        coarse_confidence_threshold=0.80,
        coarse_margin_threshold=0.15,
        minimum_confidence_for_coarse=0.35,
    )

    distribution_df = build_hierarchical_decision_distribution(
        known_decisions_df=known_decisions,
        unknown_decisions_df=unknown_decisions,
    )

    assert set(distribution_df["dataset"]) == {"known_test", "local_unknown"}
    assert set(distribution_df["decision_type"]) == {
        "fine_label",
        "coarse_label",
        "manual_review",
    }


def test_missing_probability_column_raises_error():
    prediction_df = create_known_prediction_df().drop(columns=["prob_plastic"])

    with pytest.raises(ValueError, match="missing probability columns"):
        add_hierarchical_scores(
            predictions_df=prediction_df,
            fine_to_coarse=FINE_TO_COARSE,
        )