from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_manual_review_records_completion_plan_files_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "manual_review_records_completion_plan_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "thesis"
        / "active_learning_retraining_impact_plan_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "manual_review_records_completion_summary_v1.md"
    ).exists()


def test_manual_review_plan_mentions_known_and_unknown_separation():
    text = read_text("docs/methodology/manual_review_records_completion_plan_v1.md")

    assert "known training candidate" in text
    assert "unknown test candidate" in text
    assert "future class candidate" in text
    assert "exclude_or_review_again" in text


def test_manual_review_plan_mentions_current_known_labels():
    text = read_text("docs/methodology/manual_review_records_completion_plan_v1.md")

    assert "paper_cardboard" in text
    assert "plastic" in text
    assert "glass" in text
    assert "metal" in text
    assert "organic" in text
    assert "residual" in text


def test_manual_review_plan_mentions_local_seed_example():
    text = read_text("docs/methodology/manual_review_records_completion_plan_v1.md")

    assert "local_000001" in text
    assert "rubber slipper / flip-flop" in text
    assert "unknown/future-class candidate" in text


def test_active_learning_impact_plan_mentions_before_after_comparison():
    text = read_text("docs/thesis/active_learning_retraining_impact_plan_v1.md")

    assert "Baseline C" in text
    assert "Baseline C-AL" in text
    assert "before and after active learning" in text
    assert "macro-F1" in text
    assert "unknown handling" in text


def test_supervisor_summary_mentions_retraining_condition():
    text = read_text("docs/supervisor_updates/manual_review_records_completion_summary_v1.md")

    assert "full retraining cycle" in text
    assert "known-class retraining candidates" in text
    assert "future improvement" in text