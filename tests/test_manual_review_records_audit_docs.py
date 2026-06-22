from pathlib import Path

from openwaste_hr.active_learning.manual_review_audit import (
    classify_review_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_manual_review_audit_outputs_exist_after_script_run():
    assert (
        PROJECT_ROOT
        / "ml"
        / "outputs"
        / "active_learning"
        / "manual_review_records_audit_v1.csv"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "results"
        / "manual_review_records_audit_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "manual_review_records_audit_summary_v1.md"
    ).exists()


def test_classifies_outside_taxonomy_record_as_unknown_future_candidate():
    decision, reason = classify_review_record(
        {
            "sample_id": "local_000001",
            "human_observation": "rubber slipper / flip-flop",
            "taxonomy_status": "outside_current_known_taxonomy",
            "recommended_action": "keep_as_unknown_test",
            "active_learning_v2_role": "unknown_test_and_future_class_candidate",
        }
    )

    assert decision == "unknown_or_future_candidate"
    assert "outside" in reason.lower() or "unknown" in reason.lower()


def test_classifies_known_plastic_record_as_training_candidate():
    decision, reason = classify_review_record(
        {
            "sample_id": "local_test_known_001",
            "human_observation": "plastic bottle",
            "taxonomy_status": "current_known_taxonomy",
            "reviewed_fine_label": "plastic",
            "reviewed_coarse_label": "recyclable",
            "recommended_action": "known_train_candidate",
        }
    )

    assert decision == "known_train_candidate"
    assert "retraining" in reason.lower()


def test_blocks_training_when_fine_label_is_missing():
    decision, reason = classify_review_record(
        {
            "sample_id": "local_test_missing_label_001",
            "human_observation": "uncertain object",
            "taxonomy_status": "current_known_taxonomy",
            "recommended_action": "known_train_candidate",
        }
    )

    assert decision == "review_needed"
    assert "safe known fine label" in reason.lower()


def test_audit_report_mentions_retraining_decision():
    text = read_text("docs/results/manual_review_records_audit_v1.md")

    assert "Manual Review Records Audit v1" in text
    assert "Retraining Decision" in text
    assert "retraining ready" in text
    assert "known train candidates" in text
    assert "unknown or future-class candidates" in text


def test_supervisor_summary_mentions_dataset_quality_rule():
    text = read_text("docs/supervisor_updates/manual_review_records_audit_summary_v1.md")

    assert "Manual Review Records Audit Summary v1" in text
    assert "known-class samples" in text
    assert "outside-taxonomy samples" in text
    assert "dataset quality" in text