from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_methodology_chapter_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "thesis"
        / "methodology_chapter_consolidated_v1.md"
    ).exists()


def test_methodology_summary_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "methodology_chapter_summary_v1.md"
    ).exists()


def test_methodology_chapter_mentions_core_flow():
    text = read_text("docs/thesis/methodology_chapter_consolidated_v1.md")

    assert "taxonomy design" in text
    assert "dataset preparation" in text
    assert "baseline model training" in text
    assert "hierarchical decision policy" in text
    assert "active learning" in text.lower()


def test_methodology_chapter_mentions_decision_types():
    text = read_text("docs/thesis/methodology_chapter_consolidated_v1.md")

    assert "fine_label" in text
    assert "coarse_label" in text
    assert "manual_review" in text


def test_methodology_chapter_mentions_local_unknown_and_safe_tuning():
    text = read_text("docs/thesis/methodology_chapter_consolidated_v1.md")

    assert "local unknown" in text.lower()
    assert "safe policy tuning" in text.lower()
    assert "unknown false acceptance" in text.lower()


def test_methodology_summary_mentions_future_plan():
    text = read_text("docs/supervisor_updates/methodology_chapter_summary_v1.md")

    assert "pretrained transfer-learning model" in text
    assert "additional public waste datasets" in text
    assert "reviewed local dataset v2" in text
    assert "original baseline" in text