from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_pretrained_hierarchical_config_exists():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "pretrained_hierarchical_decision_policy.yaml"
    ).exists()


def test_pretrained_hierarchical_config_points_to_pretrained_checkpoint():
    text = read_text("ml/configs/pretrained_hierarchical_decision_policy.yaml")

    assert "pretrained_trashnet_v1" in text
    assert "pretrained_trashnet_v1_best.pt" in text
    assert "pretrained_hierarchical" in text


def test_pretrained_hierarchical_config_uses_pretrained_fine_threshold():
    text = read_text("ml/configs/pretrained_hierarchical_decision_policy.yaml")

    assert (
        "fine_confidence_threshold: 0.99" in text
        or "fine_confidence_threshold: 0.9900" in text
    )


def test_pretrained_hierarchical_methodology_doc_mentions_decisions():
    text = read_text("docs/methodology/pretrained_hierarchical_policy_evaluation_v1.md")

    assert "Baseline A" in text
    assert "Baseline B" in text
    assert "fine_label" in text
    assert "coarse_label" in text
    assert "manual_review" in text
    assert "0.9900" in text


def test_pretrained_hierarchical_summary_mentions_safe_tuning_next():
    text = read_text(
        "docs/supervisor_updates/pretrained_hierarchical_policy_evaluation_summary_v1.md"
    )

    assert "pretrained transfer-learning checkpoint" in text
    assert "local unknown handling" in text
    assert "safe" in text.lower()
    assert "coverage" in text.lower()