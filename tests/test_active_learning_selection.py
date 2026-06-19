import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.evaluation.active_learning_selection import (  # noqa: E402
    add_active_learning_scores,
    build_candidate_distribution,
    compute_normalized_entropy,
    get_probability_columns,
    select_candidates_by_quota,
    summarize_candidates,
)


def create_decisions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "local_000001",
                "image_path": "local_000001.jpg",
                "pred_label": "plastic",
                "max_softmax_confidence": 0.91,
                "top_coarse_label": "recyclable",
                "top_coarse_confidence": 0.95,
                "coarse_margin": 0.90,
                "hierarchical_decision_type": "fine_label",
                "hierarchical_final_label": "plastic",
                "prob_paper_cardboard": 0.02,
                "prob_plastic": 0.91,
                "prob_glass": 0.03,
                "prob_metal": 0.02,
                "prob_residual": 0.02,
            },
            {
                "sample_id": "local_000002",
                "image_path": "local_000002.jpg",
                "pred_label": "metal",
                "max_softmax_confidence": 0.67,
                "top_coarse_label": "recyclable",
                "top_coarse_confidence": 0.88,
                "coarse_margin": 0.70,
                "hierarchical_decision_type": "coarse_label",
                "hierarchical_final_label": "recyclable",
                "prob_paper_cardboard": 0.10,
                "prob_plastic": 0.08,
                "prob_glass": 0.03,
                "prob_metal": 0.67,
                "prob_residual": 0.12,
            },
            {
                "sample_id": "local_000003",
                "image_path": "local_000003.jpg",
                "pred_label": "paper_cardboard",
                "max_softmax_confidence": 0.31,
                "top_coarse_label": "recyclable",
                "top_coarse_confidence": 0.60,
                "coarse_margin": 0.20,
                "hierarchical_decision_type": "manual_review",
                "hierarchical_final_label": "manual_review",
                "prob_paper_cardboard": 0.31,
                "prob_plastic": 0.20,
                "prob_glass": 0.15,
                "prob_metal": 0.14,
                "prob_residual": 0.20,
            },
        ]
    )


DECISION_PRIORITY_WEIGHTS = {
    "manual_review": 1.00,
    "coarse_label": 0.75,
    "fine_label": 0.60,
}


SCORE_WEIGHTS = {
    "decision_priority": 0.45,
    "entropy": 0.25,
    "confidence_uncertainty": 0.20,
    "coarse_margin_uncertainty": 0.10,
}


def test_get_probability_columns():
    df = create_decisions_df()

    probability_columns = get_probability_columns(df)

    assert "prob_plastic" in probability_columns
    assert "prob_residual" in probability_columns


def test_compute_normalized_entropy():
    confident_entropy = compute_normalized_entropy([0.90, 0.05, 0.03, 0.01, 0.01])
    uncertain_entropy = compute_normalized_entropy([0.20, 0.20, 0.20, 0.20, 0.20])

    assert confident_entropy < uncertain_entropy
    assert uncertain_entropy == 1.0


def test_add_active_learning_scores():
    df = create_decisions_df()

    scored_df = add_active_learning_scores(
        decisions_df=df,
        decision_priority_weights=DECISION_PRIORITY_WEIGHTS,
        score_weights=SCORE_WEIGHTS,
    )

    assert "active_learning_score" in scored_df.columns
    assert "active_learning_reason" in scored_df.columns
    assert scored_df.loc[2, "active_learning_reason"] == "manual_review_candidate"


def test_select_candidates_by_quota():
    df = create_decisions_df()

    scored_df = add_active_learning_scores(
        decisions_df=df,
        decision_priority_weights=DECISION_PRIORITY_WEIGHTS,
        score_weights=SCORE_WEIGHTS,
    )

    candidates_df = select_candidates_by_quota(
        scored_df=scored_df,
        total_candidates=2,
        quota_by_decision_type={
            "manual_review": 1,
            "coarse_label": 1,
            "fine_label": 1,
        },
    )

    assert len(candidates_df) == 2
    assert "candidate_rank" in candidates_df.columns
    assert candidates_df["sample_id"].is_unique


def test_build_candidate_distribution():
    df = create_decisions_df()

    scored_df = add_active_learning_scores(
        decisions_df=df,
        decision_priority_weights=DECISION_PRIORITY_WEIGHTS,
        score_weights=SCORE_WEIGHTS,
    )

    distribution_df = build_candidate_distribution(scored_df)

    assert set(distribution_df["decision_type"]) == {
        "manual_review",
        "coarse_label",
        "fine_label",
    }


def test_summarize_candidates():
    df = create_decisions_df()

    scored_df = add_active_learning_scores(
        decisions_df=df,
        decision_priority_weights=DECISION_PRIORITY_WEIGHTS,
        score_weights=SCORE_WEIGHTS,
    )

    summary = summarize_candidates(scored_df)

    assert summary["selected_candidate_count"] == 3
    assert summary["manual_review_candidates"] == 1
    assert summary["coarse_label_candidates"] == 1
    assert summary["fine_label_candidates"] == 1


def test_missing_probability_columns_raises_error():
    df = create_decisions_df().drop(
        columns=[
            "prob_paper_cardboard",
            "prob_plastic",
            "prob_glass",
            "prob_metal",
            "prob_residual",
        ]
    )

    with pytest.raises(ValueError, match="No probability columns"):
        add_active_learning_scores(
            decisions_df=df,
            decision_priority_weights=DECISION_PRIORITY_WEIGHTS,
            score_weights=SCORE_WEIGHTS,
        )