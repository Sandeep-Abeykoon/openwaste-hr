import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.evaluation.selective_metrics import (  # noqa: E402
    apply_confidence_threshold,
    compute_selective_metrics,
    select_threshold_from_sweep,
    sweep_confidence_thresholds,
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
                "max_softmax_confidence": 0.95,
            },
            {
                "sample_id": "sample_002",
                "true_label_id": 1,
                "true_label": "metal",
                "pred_label_id": 1,
                "pred_label": "metal",
                "correct": True,
                "max_softmax_confidence": 0.80,
            },
            {
                "sample_id": "sample_003",
                "true_label_id": 1,
                "true_label": "metal",
                "pred_label_id": 2,
                "pred_label": "residual",
                "correct": False,
                "max_softmax_confidence": 0.60,
            },
            {
                "sample_id": "sample_004",
                "true_label_id": 2,
                "true_label": "residual",
                "pred_label_id": 2,
                "pred_label": "residual",
                "correct": True,
                "max_softmax_confidence": 0.40,
            },
        ]
    )


def test_apply_confidence_threshold_marks_manual_review():
    prediction_df = create_prediction_df()

    thresholded_df = apply_confidence_threshold(
        predictions_df=prediction_df,
        threshold=0.70,
    )

    assert thresholded_df.loc[0, "decision"] == "accept_fine_label"
    assert thresholded_df.loc[1, "decision"] == "accept_fine_label"
    assert thresholded_df.loc[2, "decision"] == "manual_review"
    assert thresholded_df.loc[3, "decision"] == "manual_review"

    assert thresholded_df.loc[2, "final_label"] == "manual_review"
    assert thresholded_df.loc[2, "reject_reason"] == "confidence_below_threshold"


def test_compute_selective_metrics():
    prediction_df = create_prediction_df()
    thresholded_df = apply_confidence_threshold(
        predictions_df=prediction_df,
        threshold=0.70,
    )

    metrics = compute_selective_metrics(
        thresholded_df=thresholded_df,
        label_names=["plastic", "metal", "residual"],
    )

    assert metrics["total_samples"] == 4
    assert metrics["accepted_count"] == 2
    assert metrics["rejected_count"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["rejection_rate"] == 0.5
    assert metrics["forced_accuracy"] == 0.75
    assert metrics["selective_accuracy"] == 1.0
    assert metrics["selective_error_rate"] == 0.0


def test_sweep_and_select_threshold():
    prediction_df = create_prediction_df()

    sweep_df = sweep_confidence_thresholds(
        predictions_df=prediction_df,
        label_names=["plastic", "metal", "residual"],
        thresholds=[0.0, 0.5, 0.7, 0.9],
    )

    selected = select_threshold_from_sweep(
        sweep_df=sweep_df,
        min_coverage=0.5,
        selection_metric="selective_macro_f1",
    )

    assert "threshold" in selected
    assert selected["coverage"] >= 0.5


def test_invalid_threshold_raises_error():
    prediction_df = create_prediction_df()

    with pytest.raises(ValueError, match="Threshold must be between"):
        apply_confidence_threshold(
            predictions_df=prediction_df,
            threshold=1.5,
        )