from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_exists():
    assert (PROJECT_ROOT / "README.md").exists()


def test_readme_mentions_project_title_and_summary():
    text = read_text("README.md")

    assert "OpenWaste-HR" in text
    assert "Hierarchical Open-Set Waste Classification" in text
    assert "reject/manual-review" in text
    assert "human-in-the-loop active learning" in text


def test_readme_mentions_final_expanded_public_result():
    text = read_text("README.md")

    assert "Final Expanded Public Result" in text
    assert "Baseline C + Safe Policy C" in text
    assert "pretrained expanded public model" in text
    assert "expanded public safe hierarchical decision policy" in text


def test_readme_mentions_final_known_class_metrics():
    text = read_text("README.md")

    assert "Final Closed-Set Known Classification" in text
    assert "Scratch TrashNet-only baseline" in text
    assert "Pretrained TrashNet-only model" in text
    assert "Pretrained expanded public model" in text
    assert "0.8876" in text
    assert "0.8819" in text


def test_readme_mentions_reject_option_and_unknown_results():
    text = read_text("README.md")

    assert "Final Reject-Option Result" in text
    assert "Confidence threshold" in text
    assert "Max-logit" in text
    assert "Energy score" in text
    assert "Final Unknown Handling Result" in text
    assert "0.6750" in text
    assert "0.6509" in text


def test_readme_mentions_final_safe_hierarchical_policy():
    text = read_text("README.md")

    assert "Final Safe Hierarchical Policy" in text
    assert "fine confidence threshold" in text
    assert "coarse confidence threshold" in text
    assert "minimum confidence for coarse fallback" in text
    assert "0.9838" in text


def test_readme_mentions_human_review_active_learning_status():
    text = read_text("README.md")

    assert "Human Review and Active Learning Status" in text
    assert "active learning candidate selection" in text
    assert "retraining with reviewed known-class samples" in text
    assert "before/after active learning improvement comparison" in text
    assert "pending" in text


def test_readme_mentions_local_unknown_example():
    text = read_text("README.md")

    assert "Local Unknown Example" in text
    assert "local_000001" in text
    assert "rubber slipper / flip-flop" in text
    assert "outside_current_known_taxonomy" in text
    assert "unknown_test_and_future_class_candidate" in text


def test_readme_mentions_repository_structure_and_commands():
    text = read_text("README.md")

    assert "Repository Structure" in text
    assert "backend/" in text
    assert "frontend/" in text
    assert "ml/" in text
    assert "pytest" in text
    assert "uvicorn backend.app.main:app" in text


def test_readme_mentions_key_thesis_files():
    text = read_text("README.md")

    assert "Key Thesis and Result Files" in text
    assert "docs/thesis/methodology_chapter_consolidated_v1.md" in text
    assert "docs/thesis/implementation_chapter_draft_v1.md" in text
    assert "docs/results/final_expanded_public_model_policy_comparison_v1.md" in text
    assert "docs/thesis/evaluation_expanded_public_final_update_v1.md" in text
    assert "docs/thesis/final_project_summary_after_expansion_v1.md" in text
    assert "docs/thesis/active_learning_v2_section_v1.md" in text
    assert "docs/supervisor_updates/supervisor_handover_pack_v1.md" in text


def test_readme_mentions_completed_pipeline_and_remaining_work():
    text = read_text("README.md")

    assert "Completed Pipeline" in text
    assert "expanded public pretrained training" in text
    assert "expanded public safe hierarchical policy tuning" in text
    assert "Remaining Work" in text
    assert "complete manual review" in text
    assert "fine-tune the expanded public pretrained model" in text
    assert "TACO" in text


def test_readme_mentions_terminology_without_old_confusing_wording():
    text = read_text("README.md")

    assert "Terminology" in text
    assert "local unknown dataset" in text
    assert "manual_review" in text
    assert "coarse fallback" in text
    assert "expanded public safe hierarchical policy" in text
    assert "outside_current_known_taxonomy" in text

    assert "corrected unknown dataset" not in text


def test_readme_mentions_current_status_and_next_steps():
    text = read_text("README.md")

    assert "Current Status" in text
    assert "complete expanded public research and prototype pipeline" in text
    assert "human-review retraining cycle" in text
    assert "TACO dataset feasibility and intake planning" in text