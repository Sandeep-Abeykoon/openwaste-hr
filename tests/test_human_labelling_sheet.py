import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.evaluation.build_human_labelling_sheet import (  # noqa: E402
    build_human_labelling_sheet,
    build_labelling_summary,
    validate_candidate_columns,
    validate_human_labelling_sheet,
)


def create_candidates_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "sample_id": "local_000001",
                "image_path": "ml/data/local_unknown/images/local_000001.jpg",
                "object_description": "old rubber slipper",
                "pred_label": "plastic",
                "max_softmax_confidence": 0.962933,
                "hierarchical_decision_type": "fine_label",
                "hierarchical_final_label": "plastic",
                "active_learning_score": 0.306373,
                "active_learning_reason": "fine_accepted_unknown_candidate",
            },
            {
                "candidate_rank": 2,
                "sample_id": "local_000002",
                "image_path": "ml/data/local_unknown/images/local_000002.jpg",
                "object_description": "metal cap-like household object",
                "pred_label": "residual",
                "max_softmax_confidence": 0.381314,
                "hierarchical_decision_type": "manual_review",
                "hierarchical_final_label": "manual_review",
                "active_learning_score": 0.846766,
                "active_learning_reason": "manual_review_candidate",
            },
        ]
    )


def test_validate_candidate_columns():
    candidates_df = create_candidates_df()

    assert validate_candidate_columns(candidates_df)


def test_build_human_labelling_sheet_adds_annotation_columns():
    candidates_df = create_candidates_df()

    labelling_df = build_human_labelling_sheet(candidates_df)

    assert "human_decision" in labelling_df.columns
    assert "human_fine_label" in labelling_df.columns
    assert "human_coarse_label" in labelling_df.columns
    assert "proposed_new_label" in labelling_df.columns
    assert "human_confidence" in labelling_df.columns
    assert "human_notes" in labelling_df.columns


def test_validate_human_labelling_sheet():
    candidates_df = create_candidates_df()
    labelling_df = build_human_labelling_sheet(candidates_df)

    assert validate_human_labelling_sheet(labelling_df)


def test_build_labelling_summary():
    candidates_df = create_candidates_df()
    labelling_df = build_human_labelling_sheet(candidates_df)

    summary = build_labelling_summary(labelling_df)

    assert summary["total_labelling_rows"] == 2
    assert summary["manual_review_candidates"] == 1
    assert summary["fine_label_candidates"] == 1


def test_missing_candidate_column_raises_error():
    candidates_df = create_candidates_df().drop(columns=["sample_id"])

    with pytest.raises(ValueError, match="Candidate file is missing required columns"):
        validate_candidate_columns(candidates_df)