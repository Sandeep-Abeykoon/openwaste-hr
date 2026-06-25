import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.evaluation.open_set_scores import (  # noqa: E402
    apply_score_threshold,
    build_thresholds_from_scores,
    compute_energy_score,
    compute_max_logit_score,
    compute_score_selective_metrics,
    select_score_threshold_from_sweep,
    sweep_score_thresholds,
)


def create_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "true_label_id": 0,
                "true_label": "plastic",
                "pred_label_id": 0,
                "pred_label": "plastic",
                "correct": True,
                "max_logit_score": 5.0,
                "energy_score": -5.1,
            },
            {
                "sample_id": "sample_002",
                "true_label_id": 1,
                "true_label": "metal",
                "pred_label_id": 1,
                "pred_label": "metal",
                "correct": True,
                "max_logit_score": 4.0,
                "energy_score": -4.2,
            },
            {
                "sample_id": "sample_003",
                "true_label_id": 1,
                "true_label": "metal",
                "pred_label_id": 2,
                "pred_label": "residual",
                "correct": False,
                "max_logit_score": 2.0,
                "energy_score": -2.1,
            },
            {
                "sample_id": "sample_004",
                "true_label_id": 2,
                "true_label": "residual",
                "pred_label_id": 2,
                "pred_label": "residual",
                "correct": True,
                "max_logit_score": 1.0,
                "energy_score": -1.2,
            },
        ]
    )


def test_compute_max_logit_score():
    logits = torch.tensor(
        [
            [1.0, 2.0, 0.5],
            [-1.0, -0.5, -2.0],
        ]
    )

    scores = compute_max_logit_score(logits)

    assert scores.tolist() == [2.0, -0.5]


def test_compute_energy_score():
    logits = torch.tensor(
        [
            [1.0, 2.0, 0.5],
            [-1.0, -0.5, -2.0],
        ]
    )

    scores = compute_energy_score(logits, temperature=1.0)

    expected = -torch.logsumexp(logits, dim=1)

    assert torch.allclose(scores, expected)


def test_build_thresholds_from_scores():
    thresholds = build_thresholds_from_scores(
        scores=pd.Series([1.0, 2.0, 3.0]),
        threshold_count=3,
    )

    assert thresholds == [1.0, 2.0, 3.0]


def test_apply_score_threshold_greater_equal():
    prediction_df = create_prediction_df()

    thresholded_df = apply_score_threshold(
        predictions_df=prediction_df,
        score_column="max_logit_score",
        threshold=3.0,
        accept_direction="greater_equal",
    )

    assert thresholded_df.loc[0, "decision"] == "accept_fine_label"
    assert thresholded_df.loc[1, "decision"] == "accept_fine_label"
    assert thresholded_df.loc[2, "decision"] == "manual_review"


def test_apply_score_threshold_less_equal():
    prediction_df = create_prediction_df()

    thresholded_df = apply_score_threshold(
        predictions_df=prediction_df,
        score_column="energy_score",
        threshold=-3.0,
        accept_direction="less_equal",
    )

    assert thresholded_df.loc[0, "decision"] == "accept_fine_label"
    assert thresholded_df.loc[1, "decision"] == "accept_fine_label"
    assert thresholded_df.loc[2, "decision"] == "manual_review"


def test_compute_score_selective_metrics():
    prediction_df = create_prediction_df()

    thresholded_df = apply_score_threshold(
        predictions_df=prediction_df,
        score_column="max_logit_score",
        threshold=3.0,
        accept_direction="greater_equal",
    )

    metrics = compute_score_selective_metrics(
        thresholded_df=thresholded_df,
        label_names=["plastic", "metal", "residual"],
    )

    assert metrics["total_samples"] == 4
    assert metrics["accepted_count"] == 2
    assert metrics["rejected_count"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["selective_accuracy"] == 1.0


def test_sweep_and_select_score_threshold():
    prediction_df = create_prediction_df()

    sweep_df = sweep_score_thresholds(
        predictions_df=prediction_df,
        label_names=["plastic", "metal", "residual"],
        score_name="max_logit",
        score_column="max_logit_score",
        thresholds=[1.0, 3.0, 5.0],
        accept_direction="greater_equal",
    )

    selected = select_score_threshold_from_sweep(
        sweep_df=sweep_df,
        min_coverage=0.5,
        selection_metric="selective_macro_f1",
    )

    assert selected["coverage"] >= 0.5
    assert selected["score_name"] == "max_logit"


def test_invalid_accept_direction_raises_error():
    prediction_df = create_prediction_df()

    with pytest.raises(ValueError, match="accept_direction"):
        apply_score_threshold(
            predictions_df=prediction_df,
            score_column="max_logit_score",
            threshold=3.0,
            accept_direction="bad_direction",
        )