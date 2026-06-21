from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_prototype_run_guide_exists():
    assert (PROJECT_ROOT / "docs" / "methodology" / "prototype_run_guide_v1.md").exists()


def test_demo_checklist_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "prototype_demo_checklist_v1.md"
    ).exists()


def test_prototype_run_guide_mentions_backend_and_frontend():
    text = read_text("docs/methodology/prototype_run_guide_v1.md")

    assert "uvicorn backend.app.main:app" in text
    assert "start frontend\\index.html" in text
    assert "/api/inference/predict" in text
    assert "fine_label" in text
    assert "manual_review" in text


def test_demo_checklist_mentions_expected_result():
    text = read_text("docs/supervisor_updates/prototype_demo_checklist_v1.md")

    assert "110 passed" in text
    assert "local_000001" in text
    assert "plastic" in text
    assert "fine_label" in text


def test_demo_checklist_mentions_project_novelty():
    text = read_text("docs/supervisor_updates/prototype_demo_checklist_v1.md")

    assert "coarse_label" in text
    assert "manual_review" in text
    assert "active learning" in text.lower()