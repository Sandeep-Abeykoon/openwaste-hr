import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.evaluation.process_reviewed_labels import (  # noqa: E402
    build_ready_for_dataset_rows,
    build_review_summary,
    process_reviewed_labels,
    validate_required_sheet_columns,
    validate_single_review_row,
)


ANNOTATION_CONFIG = {
    "allowed_human_decisions": [
        "known_label",
        "new_unknown_class",
        "mixed_waste",
        "unclear_image",
        "remove_sample",
    ],
    "allowed_fine_labels": [
        "paper_cardboard",
        "plastic",
        "glass",
        "metal",
        "organic",
        "e_waste_battery",
        "residual",
        "unknown",
    ],
    "allowed_coarse_labels": [
        "recyclable",
        "organic",
        "hazardous",
        "residual",
        "unknown",
        "manual_review",
    ],
    "allowed_human_confidence": ["low", "medium", "high"],
}


def create_sheet_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "sample_id": "local_000001",
                "image_path": "local_000001.jpg",
                "hierarchical_decision_type": "fine_label",
                "hierarchical_final_label": "plastic",
                "active_learning_score": 0.30,
                "human_decision": "",
                "human_fine_label": "",
                "human_coarse_label": "",
                "proposed_new_label": "",
                "human_confidence": "",
                "human_notes": "",
                "reviewed_by": "",
                "review_date": "",
            },
            {
                "candidate_rank": 2,
                "sample_id": "local_000002",
                "image_path": "local_000002.jpg",
                "hierarchical_decision_type": "manual_review",
                "hierarchical_final_label": "manual_review",
                "active_learning_score": 0.84,
                "human_decision": "known_label",
                "human_fine_label": "plastic",
                "human_coarse_label": "recyclable",
                "proposed_new_label": "",
                "human_confidence": "high",
                "human_notes": "clear plastic item",
                "reviewed_by": "JB",
                "review_date": "2026-06-19",
            },
            {
                "candidate_rank": 3,
                "sample_id": "local_000003",
                "image_path": "local_000003.jpg",
                "hierarchical_decision_type": "manual_review",
                "hierarchical_final_label": "manual_review",
                "active_learning_score": 0.82,
                "human_decision": "new_unknown_class",
                "human_fine_label": "",
                "human_coarse_label": "unknown",
                "proposed_new_label": "rubber_footwear",
                "human_confidence": "medium",
                "human_notes": "new local class",
                "reviewed_by": "JB",
                "review_date": "2026-06-19",
            },
        ]
    )


def test_validate_required_sheet_columns():
    sheet_df = create_sheet_df()

    assert validate_required_sheet_columns(sheet_df)


def test_pending_review_row():
    sheet_df = create_sheet_df()
    row = sheet_df.iloc[0]

    status, message = validate_single_review_row(
        row=row,
        annotation_config=ANNOTATION_CONFIG,
    )

    assert status == "pending_review"
    assert "No human decision" in message


def test_process_reviewed_labels():
    sheet_df = create_sheet_df()

    processed_df = process_reviewed_labels(
        sheet_df=sheet_df,
        annotation_config=ANNOTATION_CONFIG,
    )

    assert "review_status" in processed_df.columns
    assert "dataset_action" in processed_df.columns
    assert processed_df.loc[0, "review_status"] == "pending_review"
    assert processed_df.loc[1, "dataset_action"] == "add_as_known_sample"
    assert processed_df.loc[2, "dataset_action"] == "add_as_new_unknown_candidate"


def test_build_ready_for_dataset_rows():
    sheet_df = create_sheet_df()

    processed_df = process_reviewed_labels(
        sheet_df=sheet_df,
        annotation_config=ANNOTATION_CONFIG,
    )

    ready_df = build_ready_for_dataset_rows(processed_df)

    assert len(ready_df) == 2
    assert set(ready_df["dataset_action"]) == {
        "add_as_known_sample",
        "add_as_new_unknown_candidate",
    }


def test_build_review_summary():
    sheet_df = create_sheet_df()

    processed_df = process_reviewed_labels(
        sheet_df=sheet_df,
        annotation_config=ANNOTATION_CONFIG,
    )

    summary = build_review_summary(processed_df)

    assert summary["total_rows"] == 3
    assert summary["reviewed_rows"] == 2
    assert summary["pending_review_rows"] == 1
    assert summary["ready_for_dataset_rows"] == 2


def test_invalid_known_label_missing_fine_label():
    sheet_df = create_sheet_df()
    row = sheet_df.iloc[1].copy()
    row["human_fine_label"] = ""

    status, message = validate_single_review_row(
        row=row,
        annotation_config=ANNOTATION_CONFIG,
    )

    assert status == "invalid_review"
    assert "requires human_fine_label" in message


def test_missing_required_column_raises_error():
    sheet_df = create_sheet_df().drop(columns=["sample_id"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_required_sheet_columns(sheet_df)