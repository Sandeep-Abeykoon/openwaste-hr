import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.evaluation.unknown_metrics import (  # noqa: E402
    apply_unknown_decision_rule,
    build_accepted_label_distribution,
    compute_unknown_rejection_metrics,
)


def create_unknown_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "local_000001",
                "image_path": "ml/data/local_unknown/images/local_000001.jpg",
                "pred_label": "plastic",
                "max_softmax_confidence": 0.80,
                "max_logit_score": 3.0,
                "energy_score": -3.5,
            },
            {
                "sample_id": "local_000002",
                "image_path": "ml/data/local_unknown/images/local_000002.jpg",
                "pred_label": "metal",
                "max_softmax_confidence": 0.30,
                "max_logit_score": 1.0,
                "energy_score": -1.1,
            },
        ]
    )


def test_apply_unknown_decision_rule_confidence():
    prediction_df = create_unknown_prediction_df()

    decisions_df = apply_unknown_decision_rule(
        predictions_df=prediction_df,
        method_name="confidence",
        score_column="max_softmax_confidence",
        threshold=0.64,
        accept_direction="greater_equal",
    )

    assert decisions_df.loc[0, "final_decision"] == "accepted_as_known"
    assert decisions_df.loc[0, "unknown_eval_outcome"] == "false_accept_unknown_as_known"

    assert decisions_df.loc[1, "final_decision"] == "manual_review"
    assert decisions_df.loc[1, "unknown_eval_outcome"] == "correct_reject_unknown"


def test_compute_unknown_rejection_metrics():
    prediction_df = create_unknown_prediction_df()

    decisions_df = apply_unknown_decision_rule(
        predictions_df=prediction_df,
        method_name="confidence",
        score_column="max_softmax_confidence",
        threshold=0.64,
        accept_direction="greater_equal",
    )

    metrics = compute_unknown_rejection_metrics(decisions_df)

    assert metrics["total_unknown_samples"] == 2
    assert metrics["rejected_unknown_count"] == 1
    assert metrics["accepted_unknown_as_known_count"] == 1
    assert metrics["unknown_rejection_rate"] == 0.5
    assert metrics["unknown_false_acceptance_rate"] == 0.5


def test_build_accepted_label_distribution():
    prediction_df = create_unknown_prediction_df()

    decisions_df = apply_unknown_decision_rule(
        predictions_df=prediction_df,
        method_name="confidence",
        score_column="max_softmax_confidence",
        threshold=0.64,
        accept_direction="greater_equal",
    )

    distribution_df = build_accepted_label_distribution(decisions_df)

    assert len(distribution_df) == 1
    assert distribution_df.loc[0, "method_name"] == "confidence"
    assert distribution_df.loc[0, "pred_label"] == "plastic"
    assert distribution_df.loc[0, "accepted_count"] == 1


def test_apply_unknown_decision_rule_rejects_bad_direction():
    prediction_df = create_unknown_prediction_df()

    with pytest.raises(ValueError, match="accept_direction"):
        apply_unknown_decision_rule(
            predictions_df=prediction_df,
            method_name="bad",
            score_column="max_softmax_confidence",
            threshold=0.5,
            accept_direction="bad_direction",
        )