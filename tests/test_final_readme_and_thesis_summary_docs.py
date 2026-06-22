from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_final_readme_and_thesis_summary_files_exist():
    assert (PROJECT_ROOT / "README.md").exists()
    assert (
        PROJECT_ROOT
        / "docs"
        / "thesis"
        / "final_project_summary_after_expansion_v1.md"
    ).exists()
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "final_project_completion_summary_v1.md"
    ).exists()


def test_readme_mentions_final_expanded_public_result():
    text = read_text("README.md")

    assert "Final Expanded Public Result" in text
    assert "Baseline C + Safe Policy C" in text
    assert "pretrained expanded public model" in text
    assert "Energy score" in text


def test_thesis_summary_mentions_final_outputs():
    text = read_text("docs/thesis/final_project_summary_after_expansion_v1.md")

    assert "fine label" in text
    assert "coarse fallback" in text
    assert "manual review" in text
    assert "RealWaste Textile Trash" in text


def test_thesis_summary_mentions_key_final_metrics():
    text = read_text("docs/thesis/final_project_summary_after_expansion_v1.md")

    assert "0.8819" in text
    assert "0.9838" in text
    assert "0.6750" in text
    assert "0.6509" in text


def test_supervisor_completion_summary_mentions_pipeline_scope():
    text = read_text("docs/supervisor_updates/final_project_completion_summary_v1.md")

    assert "expanded public pretrained training" in text
    assert "reject-option evaluation" in text
    assert "public unknown/future-class evaluation" in text
    assert "safe hierarchical policy tuning" in text


def test_supervisor_completion_summary_mentions_recommended_thesis_position():
    text = read_text("docs/supervisor_updates/final_project_completion_summary_v1.md")

    assert "hierarchical uncertainty-aware waste classification system" in text
    assert "fine label" in text
    assert "coarse category" in text
    assert "manual review" in text