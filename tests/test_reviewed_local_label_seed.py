import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_reviewed_local_label_seed_docs_exist():
    assert (
        PROJECT_ROOT / "docs" / "methodology" / "reviewed_local_label_seed_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "reviewed_local_label_seed_summary_v1.md"
    ).exists()


def test_reviewed_local_label_seed_script_exists():
    assert (
        PROJECT_ROOT
        / "ml"
        / "src"
        / "openwaste_hr"
        / "evaluation"
        / "process_reviewed_local_label_seed.py"
    ).exists()


def test_reviewed_local_label_seed_outputs_exist_after_processing():
    assert (
        PROJECT_ROOT
        / "ml"
        / "data"
        / "splits"
        / "reviewed_local_labels_seed_v1.csv"
    ).exists()

    assert (
        PROJECT_ROOT
        / "ml"
        / "outputs"
        / "metrics"
        / "reviewed_local_labels_seed_summary_v1.json"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "results"
        / "reviewed_local_labels_seed_v1_report.md"
    ).exists()


def test_reviewed_local_label_seed_csv_records_rubber_slipper():
    path = (
        PROJECT_ROOT
        / "ml"
        / "data"
        / "splits"
        / "reviewed_local_labels_seed_v1.csv"
    )
    df = pd.read_csv(path)

    assert len(df) == 1
    assert df.loc[0, "sample_id"] == "local_000001"
    assert "rubber slipper" in df.loc[0, "human_observed_object"]
    assert df.loc[0, "human_taxonomy_status"] == "outside_current_known_taxonomy"
    assert df.loc[0, "recommended_action"] == "keep_as_unknown_test"
    assert df.loc[0, "proposed_new_label"] == "rubber_slipper_flip_flop"


def test_reviewed_local_label_seed_summary_has_expected_counts():
    path = (
        PROJECT_ROOT
        / "ml"
        / "outputs"
        / "metrics"
        / "reviewed_local_labels_seed_summary_v1.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["total_sheet_rows"] == 20
    assert data["seeded_review_rows"] == 1
    assert data["pending_review_rows"] == 19
    assert data["ready_for_active_learning_v2_rows"] == 1


def test_seeded_working_review_sheet_contains_local_000001_human_fields():
    path = (
        PROJECT_ROOT
        / "ml"
        / "outputs"
        / "metrics"
        / "human_labelling_sheet_v1_seeded_review.csv"
    )
    df = pd.read_csv(path)
    row = df.loc[df["sample_id"] == "local_000001"].iloc[0]

    assert row["human_decision"] == "outside_current_known_taxonomy"
    assert row["proposed_new_label"] == "rubber_slipper_flip_flop"
    assert row["recommended_human_action"] == "keep_as_unknown_test"
    assert "rubber slipper" in row["human_notes"]


def test_reviewed_local_label_seed_docs_mention_active_learning_v2():
    methodology_text = read_text("docs/methodology/reviewed_local_label_seed_v1.md")
    summary_text = read_text(
        "docs/supervisor_updates/reviewed_local_label_seed_summary_v1.md"
    )

    assert "active learning v2" in methodology_text.lower()
    assert "rubber slipper" in methodology_text
    assert "outside_current_known_taxonomy" in methodology_text
    assert "human-in-the-loop active learning" in summary_text