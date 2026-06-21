from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_thesis_assembly_checklist_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "thesis"
        / "thesis_assembly_checklist_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "thesis_assembly_checklist_summary_v1.md"
    ).exists()


def test_thesis_assembly_checklist_mentions_best_message():
    text = read_text("docs/thesis/thesis_assembly_checklist_v1.md")

    assert "pretrained image recognition" in text
    assert "hierarchical open-set decision-making" in text
    assert "local unknown evaluation" in text
    assert "human-in-the-loop active learning" in text


def test_thesis_assembly_checklist_mentions_chapters():
    text = read_text("docs/thesis/thesis_assembly_checklist_v1.md")

    assert "Chapter 1: Introduction" in text
    assert "Chapter 3: Methodology" in text
    assert "Chapter 4: Implementation" in text
    assert "Chapter 5: Evaluation and Results" in text
    assert "Chapter 7: Conclusion and Future Work" in text


def test_thesis_assembly_checklist_mentions_best_policy_metrics():
    text = read_text("docs/thesis/thesis_assembly_checklist_v1.md")

    assert "pretrained safe hierarchical policy" in text
    assert "0.864583" in text
    assert "0.960843" in text
    assert "0.600000" in text
    assert "0.400000" in text


def test_thesis_assembly_checklist_mentions_local_unknown_example():
    text = read_text("docs/thesis/thesis_assembly_checklist_v1.md")

    assert "local_000001" in text
    assert "rubber slipper" in text
    assert "flip-flop" in text
    assert "outside_current_known_taxonomy" in text
    assert "unknown_test_and_future_class_candidate" in text


def test_thesis_assembly_checklist_mentions_terminology_warning():
    text = read_text("docs/thesis/thesis_assembly_checklist_v1.md")

    assert "local unknown dataset" in text
    assert "manual_review" in text
    assert "coarse fallback" in text
    assert "outside_current_known_taxonomy" in text


def test_supervisor_summary_mentions_next_stage():
    text = read_text("docs/supervisor_updates/thesis_assembly_checklist_summary_v1.md")

    assert "final checklist" in text
    assert "supervisor handover pack" in text
    assert "final dissertation insertion plan" in text