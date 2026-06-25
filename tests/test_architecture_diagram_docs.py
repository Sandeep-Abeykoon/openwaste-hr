from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_architecture_diagram_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "architecture"
        / "openwaste_hr_architecture_v1.mmd"
    ).exists()


def test_architecture_notes_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "architecture"
        / "openwaste_hr_architecture_notes_v1.md"
    ).exists()


def test_architecture_diagram_mentions_core_layers():
    text = read_text("docs/architecture/openwaste_hr_architecture_v1.mmd")

    assert "Dataset and Data Management" in text
    assert "Model Training and Evaluation" in text
    assert "Hierarchical Decision Layer" in text
    assert "Local Active Learning Loop" in text
    assert "Backend Layer" in text
    assert "Frontend Demo" in text


def test_architecture_diagram_mentions_decision_types():
    text = read_text("docs/architecture/openwaste_hr_architecture_v1.mmd")

    assert "fine_label" in text
    assert "coarse_label" in text
    assert "manual_review" in text


def test_architecture_notes_mentions_thesis_use():
    text = read_text("docs/architecture/openwaste_hr_architecture_notes_v1.md")

    assert "thesis implementation chapter" in text
    assert "OpenWaste-HR prototype architecture" in text
    assert "active learning" in text.lower()
    assert "FastAPI" in text