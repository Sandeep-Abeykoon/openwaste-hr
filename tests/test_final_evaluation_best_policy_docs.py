from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_final_evaluation_best_policy_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "thesis"
        / "evaluation_best_policy_active_learning_update_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "results"
        / "final_evaluation_summary_best_policy_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "final_evaluation_best_policy_summary_v1.md"
    ).exists()


def test_final_evaluation_update_mentions_best_policy_metrics():
    text = read_text("docs/thesis/evaluation_best_policy_active_learning_update_v1.md")

    assert "Pretrained Safe Hierarchical Policy" in text
    assert "0.864583" in text
    assert "0.960843" in text
    assert "0.600000" in text
    assert "0.400000" in text


def test_final_evaluation_update_mentions_live_demo_example():
    text = read_text("docs/thesis/evaluation_best_policy_active_learning_update_v1.md")

    assert "local_000001" in text
    assert "rubber slipper" in text
    assert "flip-flop" in text
    assert "paper_cardboard" in text
    assert "coarse_label" in text


def test_final_evaluation_update_mentions_active_learning_v2_roles():
    text = read_text("docs/thesis/evaluation_best_policy_active_learning_update_v1.md")

    assert "outside_current_known_taxonomy" in text
    assert "keep_as_unknown_test" in text
    assert "unknown_test_and_future_class_candidate" in text
    assert "Include in known training v2" in text


def test_final_evaluation_summary_mentions_policy_comparison():
    text = read_text("docs/results/final_evaluation_summary_best_policy_v1.md")

    assert "Scratch safe hierarchical" in text
    assert "Pretrained hierarchical v1" in text
    assert "Pretrained safe hierarchical" in text
    assert "0.075000" in text
    assert "0.925000" in text


def test_supervisor_summary_mentions_thesis_message():
    text = read_text("docs/supervisor_updates/final_evaluation_best_policy_summary_v1.md")

    assert "pretrained image recognition" in text
    assert "hierarchical open-set decision-making" in text
    assert "local unknown evaluation" in text
    assert "human-in-the-loop active learning" in text