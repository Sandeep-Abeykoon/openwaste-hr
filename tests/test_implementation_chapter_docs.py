from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_implementation_chapter_draft_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "thesis"
        / "implementation_chapter_draft_v1.md"
    ).exists()


def test_implementation_chapter_summary_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "implementation_chapter_summary_v1.md"
    ).exists()


def test_implementation_chapter_mentions_core_components():
    text = read_text("docs/thesis/implementation_chapter_draft_v1.md")

    assert "OpenWaste-HR" in text
    assert "hierarchical open-set waste classification" in text
    assert "active learning" in text.lower()
    assert "FastAPI" in text
    assert "frontend" in text.lower()


def test_implementation_chapter_mentions_decision_types():
    text = read_text("docs/thesis/implementation_chapter_draft_v1.md")

    assert "fine_label" in text
    assert "coarse_label" in text
    assert "manual_review" in text


def test_implementation_chapter_mentions_results():
    text = read_text("docs/thesis/implementation_chapter_draft_v1.md")

    assert "0.692708" in text
    assert "0.375000" in text
    assert "0.962933" in text
    assert "121 tests" in text


def test_supervisor_summary_mentions_novelty():
    text = read_text("docs/supervisor_updates/implementation_chapter_summary_v1.md")

    assert "Hierarchical open-set waste classification" in text
    assert "reject option" in text
    assert "local active learning" in text