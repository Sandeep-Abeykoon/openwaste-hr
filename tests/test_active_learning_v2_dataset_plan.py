import json
from pathlib import Path

import pandas as pd

from openwaste_hr.evaluation.build_active_learning_v2_dataset_plan import (
    decide_dataset_role,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_active_learning_v2_dataset_plan_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "active_learning_v2_dataset_plan_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "active_learning_v2_dataset_plan_summary_v1.md"
    ).exists()


def test_active_learning_v2_dataset_plan_script_exists():
    assert (
        PROJECT_ROOT
        / "ml"
        / "src"
        / "openwaste_hr"
        / "evaluation"
        / "build_active_learning_v2_dataset_plan.py"
    ).exists()


def test_decide_dataset_role_for_outside_taxonomy():
    row = pd.Series(
        {
            "human_taxonomy_status": "outside_current_known_taxonomy",
            "recommended_action": "keep_as_unknown_test",
            "proposed_new_label": "rubber_slipper_flip_flop",
        }
    )

    decision = decide_dataset_role(row)

    assert decision["include_in_known_training_v2"] is False
    assert decision["include_in_unknown_test_v2"] is True
    assert decision["include_as_future_class_candidate"] is True
    assert decision["active_learning_v2_role"] == "unknown_test_and_future_class_candidate"


def test_active_learning_v2_outputs_exist_after_processing():
    assert (
        PROJECT_ROOT
        / "ml"
        / "data"
        / "splits"
        / "active_learning_v2_dataset_plan_v1.csv"
    ).exists()

    assert (
        PROJECT_ROOT
        / "ml"
        / "outputs"
        / "metrics"
        / "active_learning_v2_dataset_plan_summary_v1.json"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "results"
        / "active_learning_v2_dataset_plan_v1_report.md"
    ).exists()


def test_active_learning_v2_plan_records_local_000001_correctly():
    path = (
        PROJECT_ROOT
        / "ml"
        / "data"
        / "splits"
        / "active_learning_v2_dataset_plan_v1.csv"
    )

    df = pd.read_csv(path)

    assert len(df) == 1
    assert df.loc[0, "sample_id"] == "local_000001"
    assert "rubber slipper" in df.loc[0, "human_observed_object"]
    assert df.loc[0, "include_in_known_training_v2"] == False
    assert df.loc[0, "include_in_unknown_test_v2"] == True
    assert df.loc[0, "include_as_future_class_candidate"] == True
    assert (
        df.loc[0, "active_learning_v2_role"]
        == "unknown_test_and_future_class_candidate"
    )


def test_active_learning_v2_summary_has_expected_counts():
    path = (
        PROJECT_ROOT
        / "ml"
        / "outputs"
        / "metrics"
        / "active_learning_v2_dataset_plan_summary_v1.json"
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["total_reviewed_seed_rows"] == 1
    assert data["known_training_candidates"] == 0
    assert data["unknown_test_candidates"] == 1
    assert data["future_class_candidates"] == 1
    assert data["recollection_candidates"] == 0
    assert data["excluded_candidates"] == 0


def test_active_learning_v2_docs_mention_dataset_roles():
    methodology_text = read_text("docs/methodology/active_learning_v2_dataset_plan_v1.md")
    summary_text = read_text(
        "docs/supervisor_updates/active_learning_v2_dataset_plan_summary_v1.md"
    )

    assert "active learning v2" in methodology_text.lower()
    assert "unknown_test_and_future_class_candidate" in methodology_text
    assert "rubber slipper" in summary_text
    assert "known training sample" in summary_text