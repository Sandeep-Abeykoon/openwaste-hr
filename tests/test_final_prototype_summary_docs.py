from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_final_prototype_summary_exists():
    assert (PROJECT_ROOT / "docs" / "results" / "final_prototype_summary_v1.md").exists()


def test_supervisor_final_summary_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "final_prototype_summary_v1.md"
    ).exists()


def test_final_summary_mentions_core_pipeline():
    text = read_text("docs/results/final_prototype_summary_v1.md")

    assert "dataset preparation" in text
    assert "hierarchical decision policy" in text
    assert "active learning" in text.lower()
    assert "FastAPI backend endpoint" in text
    assert "simple frontend demo" in text


def test_final_summary_mentions_decision_types():
    text = read_text("docs/results/final_prototype_summary_v1.md")

    assert "fine_label" in text
    assert "coarse_label" in text
    assert "manual_review" in text


def test_final_summary_mentions_demo_result():
    text = read_text("docs/results/final_prototype_summary_v1.md")

    assert "local_000001" in text
    assert "plastic" in text
    assert "0.962933" in text
    assert "fine_confidence_high" in text


def test_supervisor_summary_mentions_project_novelty():
    text = read_text("docs/supervisor_updates/final_prototype_summary_v1.md")

    assert "hierarchical open-set waste classification" in text
    assert "reject option" in text
    assert "local active learning" in text