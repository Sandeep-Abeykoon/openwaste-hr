from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_final_model_policy_comparison_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "results"
        / "final_model_policy_comparison_v1.md"
    ).exists()


def test_final_model_policy_supervisor_summary_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "final_model_policy_comparison_summary_v1.md"
    ).exists()


def test_final_model_policy_comparison_mentions_best_policy():
    text = read_text("docs/results/final_model_policy_comparison_v1.md")

    assert "Pretrained Safe Hierarchical Policy" in text
    assert "best current OpenWaste-HR policy" in text
    assert "0.864583" in text
    assert "0.960843" in text
    assert "0.600000" in text


def test_final_model_policy_comparison_mentions_baseline_improvement():
    text = read_text("docs/results/final_model_policy_comparison_v1.md")

    assert "Baseline A" in text
    assert "Baseline B" in text
    assert "0.692708" in text
    assert "0.888000" in text
    assert "+0.195292" in text


def test_final_model_policy_comparison_mentions_remaining_work():
    text = read_text("docs/results/final_model_policy_comparison_v1.md")

    assert "human correction labels" in text
    assert "reviewed local dataset v2" in text
    assert "additional public waste datasets" in text
    assert "active-learning feedback" in text


def test_final_model_policy_supervisor_summary_mentions_thesis_message():
    text = read_text(
        "docs/supervisor_updates/final_model_policy_comparison_summary_v1.md"
    )

    assert "pretrained image recognition" in text
    assert "hierarchical open-set decision-making" in text
    assert "manual-review routing" in text
    assert "active learning support" in text