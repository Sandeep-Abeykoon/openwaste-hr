from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_active_learning_v2_thesis_section_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "thesis"
        / "active_learning_v2_section_v1.md"
    ).exists()


def test_active_learning_v2_supervisor_summary_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "active_learning_v2_thesis_section_summary_v1.md"
    ).exists()


def test_active_learning_v2_thesis_section_mentions_reviewed_seed():
    text = read_text("docs/thesis/active_learning_v2_section_v1.md")

    assert "local_000001" in text
    assert "rubber slipper" in text
    assert "flip-flop" in text
    assert "outside_current_known_taxonomy" in text
    assert "rubber_slipper_flip_flop" in text


def test_active_learning_v2_thesis_section_mentions_dataset_roles():
    text = read_text("docs/thesis/active_learning_v2_section_v1.md")

    assert "include_in_known_training_v2" in text
    assert "include_in_unknown_test_v2" in text
    assert "include_as_future_class_candidate" in text
    assert "unknown_test_and_future_class_candidate" in text


def test_active_learning_v2_thesis_section_mentions_research_importance():
    text = read_text("docs/thesis/active_learning_v2_section_v1.md")

    assert "closed-set confidence alone is not sufficient" in text
    assert "human-in-the-loop" in text
    assert "future taxonomy expansion" in text
    assert "hierarchical open-set waste classification" in text


def test_active_learning_v2_supervisor_summary_mentions_contribution_link():
    text = read_text(
        "docs/supervisor_updates/active_learning_v2_thesis_section_summary_v1.md"
    )

    assert "local unknown evaluation" in text
    assert "manual-review routing" in text
    assert "human-in-the-loop active learning" in text
    assert "future taxonomy expansion" in text