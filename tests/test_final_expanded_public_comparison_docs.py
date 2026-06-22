from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_final_expanded_public_comparison_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "results"
        / "final_expanded_public_model_policy_comparison_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "thesis"
        / "evaluation_expanded_public_final_update_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "final_expanded_public_comparison_summary_v1.md"
    ).exists()


def test_final_expanded_public_comparison_mentions_core_systems():
    text = read_text("docs/results/final_expanded_public_model_policy_comparison_v1.md")

    assert "Baseline A" in text
    assert "Baseline B" in text
    assert "Baseline C" in text
    assert "Safe Policy B" in text
    assert "Safe Policy C" in text


def test_final_expanded_public_comparison_mentions_key_metrics():
    text = read_text("docs/results/final_expanded_public_model_policy_comparison_v1.md")

    assert "0.8819" in text
    assert "0.9732" in text
    assert "0.6750" in text
    assert "0.6509" in text
    assert "0.9838" in text


def test_final_expanded_public_comparison_mentions_tradeoff():
    text = read_text("docs/results/final_expanded_public_model_policy_comparison_v1.md")

    assert "trade-off" in text
    assert "Energy-score rejection" in text
    assert "confidence threshold" in text
    assert "local unknown" in text


def test_thesis_update_mentions_unknown_and_hierarchical_outputs():
    text = read_text("docs/thesis/evaluation_expanded_public_final_update_v1.md")

    assert "expanded public pretrained model" in text
    assert "fine labels" in text
    assert "coarse fallback" in text
    assert "manual review" in text
    assert "unknown rejection" in text


def test_supervisor_summary_mentions_final_position():
    text = read_text("docs/supervisor_updates/final_expanded_public_comparison_summary_v1.md")

    assert "strongest balanced system" in text
    assert "Energy score" in text
    assert "Known Coverage" in text
    assert "Accepted Success Rate" in text