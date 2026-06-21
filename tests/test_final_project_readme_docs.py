from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_exists_and_mentions_project_title():
    text = read_text("README.md")

    assert "OpenWaste-HR" in text
    assert "Hierarchical Open-Set Waste Classification" in text
    assert "Local Active Learning" in text


def test_readme_mentions_best_policy_and_metrics():
    text = read_text("README.md")

    assert "Pretrained Safe Hierarchical Policy" in text
    assert "0.864583" in text
    assert "0.960843" in text
    assert "0.600000" in text
    assert "0.400000" in text


def test_readme_mentions_local_unknown_example():
    text = read_text("README.md")

    assert "local_000001" in text
    assert "rubber slipper" in text
    assert "flip-flop" in text
    assert "paper_cardboard" in text
    assert "outside_current_known_taxonomy" in text


def test_readme_mentions_active_learning_v2():
    text = read_text("README.md")

    assert "Active Learning v2" in text
    assert "keep_as_unknown_test" in text
    assert "unknown_test_and_future_class_candidate" in text
    assert "human-in-the-loop active learning" in text


def test_readme_mentions_demo_commands():
    text = read_text("README.md")

    assert "uvicorn backend.app.main:app" in text
    assert "python -m http.server 5500" in text
    assert "http://127.0.0.1:5500" in text
    assert "Invoke-RestMethod" in text


def test_readme_mentions_thesis_files_and_handover_pack():
    text = read_text("README.md")

    assert "docs/thesis/methodology_chapter_consolidated_v1.md" in text
    assert "docs/thesis/implementation_chapter_draft_v1.md" in text
    assert "docs/thesis/evaluation_best_policy_active_learning_update_v1.md" in text
    assert "docs/supervisor_updates/supervisor_handover_pack_v1.md" in text


def test_readme_mentions_terminology_warning():
    text = read_text("README.md")

    assert "local unknown dataset" in text
    assert "manual_review" in text
    assert "coarse fallback" in text
    assert "corrected unknown dataset" in text


def test_final_readme_update_summary_exists_and_mentions_purpose():
    text = read_text("docs/supervisor_updates/final_readme_update_summary_v1.md")

    assert "final-year research project" in text
    assert "working prototype" in text
    assert "Pretrained Safe Hierarchical Policy" in text
    assert "complete research and prototype pipeline" in text