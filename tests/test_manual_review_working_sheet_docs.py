from pathlib import Path

from openwaste_hr.active_learning.prepare_manual_review_working_sheet import (
    infer_review_status,
    make_working_sheet_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_manual_review_working_sheet_outputs_exist_after_script_run():
    assert (
        PROJECT_ROOT
        / "ml"
        / "outputs"
        / "active_learning"
        / "manual_review_working_sheet_v1.csv"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "results"
        / "manual_review_working_sheet_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "manual_review_working_sheet_summary_v1.md"
    ).exists()


def test_pending_row_is_marked_pending_review():
    status = infer_review_status(
        {
            "sample_id": "local_pending_001",
            "image_path": "ml/data/local_unknown/images/local_pending_001.jpg",
        }
    )

    assert status == "pending_review"


def test_reviewed_row_is_marked_already_reviewed():
    status = infer_review_status(
        {
            "sample_id": "local_000001",
            "human_observation": "rubber slipper / flip-flop",
            "taxonomy_status": "outside_current_known_taxonomy",
            "recommended_action": "unknown_test_candidate",
        }
    )

    assert status == "already_reviewed"


def test_make_working_sheet_row_preserves_model_and_review_fields():
    source_file = PROJECT_ROOT / "ml" / "data" / "splits" / "dummy_candidates.csv"

    row = make_working_sheet_row(
        PROJECT_ROOT,
        source_file,
        {
            "sample_id": "candidate_001",
            "image_path": "ml/data/local_unknown/images/candidate_001.jpg",
            "predicted_label": "plastic",
            "confidence": "0.55",
            "decision_type": "manual_review",
            "selection_reason": "low confidence candidate",
            "human_observation": "plastic bottle",
            "taxonomy_status": "current_known_taxonomy",
            "reviewed_fine_label": "plastic",
            "reviewed_coarse_label": "recyclable",
            "recommended_action": "known_train_candidate",
        },
    )

    assert row["sample_id"] == "candidate_001"
    assert row["current_model_label"] == "plastic"
    assert row["current_model_confidence"] == "0.55"
    assert row["current_model_decision_type"] == "manual_review"
    assert row["human_observation"] == "plastic bottle"
    assert row["recommended_action"] == "known_train_candidate"
    assert row["review_status"] == "already_reviewed"


def test_working_sheet_report_mentions_review_fields_and_rules():
    text = read_text("docs/results/manual_review_working_sheet_v1.md")

    assert "Manual Review Working Sheet v1" in text
    assert "human_observation" in text
    assert "taxonomy_status" in text
    assert "known_train_candidate" in text
    assert "future_class_candidate" in text
    assert "Do not force outside-taxonomy samples into known classes" in text


def test_supervisor_summary_mentions_active_learning_impact_condition():
    text = read_text("docs/supervisor_updates/manual_review_working_sheet_summary_v1.md")

    assert "Manual Review Working Sheet Summary v1" in text
    assert "valid known-class retraining samples" in text
    assert "unknown or future-class samples" in text
    assert "impact of active learning" in text