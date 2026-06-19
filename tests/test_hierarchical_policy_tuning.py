import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.evaluation.tune_hierarchical_policy import (  # noqa: E402
    compute_objective_score,
    generate_threshold_grid,
    is_policy_valid,
    select_best_policy,
)


def test_generate_threshold_grid():
    search_space = {
        "fine_confidence_thresholds": [0.64, 0.80],
        "coarse_confidence_thresholds": [0.80],
        "coarse_margin_thresholds": [0.15, 0.50],
        "minimum_confidence_for_coarse_values": [0.35],
    }

    grid = generate_threshold_grid(search_space)

    assert len(grid) == 4
    assert grid[0]["fine_confidence_threshold"] == 0.64
    assert grid[-1]["coarse_margin_threshold"] == 0.50


def test_compute_objective_score():
    known_metrics = {
        "known_decision_coverage": 0.90,
        "hierarchical_success_rate_over_accepted": 0.80,
    }

    unknown_metrics = {
        "unknown_manual_review_rate": 0.50,
    }

    objective_weights = {
        "unknown_manual_review_rate": 0.50,
        "known_success_rate_over_accepted": 0.30,
        "known_decision_coverage": 0.20,
    }

    score = compute_objective_score(
        known_metrics=known_metrics,
        unknown_metrics=unknown_metrics,
        objective_weights=objective_weights,
    )

    assert score == 0.67


def test_is_policy_valid():
    known_metrics = {
        "known_decision_coverage": 0.90,
        "hierarchical_success_rate_over_accepted": 0.80,
    }

    selection_config = {
        "minimum_known_decision_coverage": 0.65,
        "minimum_known_success_rate_over_accepted": 0.75,
    }

    assert is_policy_valid(
        known_metrics=known_metrics,
        selection_config=selection_config,
    )


def test_is_policy_invalid_when_coverage_is_too_low():
    known_metrics = {
        "known_decision_coverage": 0.50,
        "hierarchical_success_rate_over_accepted": 0.90,
    }

    selection_config = {
        "minimum_known_decision_coverage": 0.65,
        "minimum_known_success_rate_over_accepted": 0.75,
    }

    assert not is_policy_valid(
        known_metrics=known_metrics,
        selection_config=selection_config,
    )


def test_select_best_policy_prefers_valid_high_score():
    sweep_df = pd.DataFrame(
        [
            {
                "fine_confidence_threshold": 0.64,
                "coarse_confidence_threshold": 0.80,
                "coarse_margin_threshold": 0.15,
                "minimum_confidence_for_coarse": 0.35,
                "is_valid_policy": True,
                "objective_score": 0.60,
                "unknown_manual_review_rate": 0.20,
                "hierarchical_success_rate_over_accepted": 0.80,
                "known_decision_coverage": 0.90,
            },
            {
                "fine_confidence_threshold": 0.80,
                "coarse_confidence_threshold": 0.95,
                "coarse_margin_threshold": 0.65,
                "minimum_confidence_for_coarse": 0.65,
                "is_valid_policy": True,
                "objective_score": 0.75,
                "unknown_manual_review_rate": 0.50,
                "hierarchical_success_rate_over_accepted": 0.78,
                "known_decision_coverage": 0.70,
            },
        ]
    )

    best_policy = select_best_policy(sweep_df)

    assert best_policy["objective_score"] == 0.75
    assert best_policy["fine_confidence_threshold"] == 0.80