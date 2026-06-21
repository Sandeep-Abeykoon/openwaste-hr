from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_pretrained_safe_hierarchical_config_exists():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "pretrained_safe_hierarchical_policy_tuning.yaml"
    ).exists()


def test_pretrained_safe_hierarchical_config_points_to_pretrained_checkpoint():
    text = read_text("ml/configs/pretrained_safe_hierarchical_policy_tuning.yaml")

    assert "pretrained_trashnet_v1" in text
    assert "pretrained_trashnet_v1_best.pt" in text
    assert "pretrained_safe_hierarchical" in text


def test_pretrained_safe_hierarchical_config_contains_strict_threshold_search():
    text = read_text("ml/configs/pretrained_safe_hierarchical_policy_tuning.yaml")

    assert "0.995" in text
    assert "0.99" in text


def test_pretrained_safe_hierarchical_methodology_doc_mentions_tradeoff():
    text = read_text(
        "docs/methodology/pretrained_safe_hierarchical_policy_tuning_v1.md"
    )

    assert "Baseline A" in text
    assert "Baseline B" in text
    assert "local unknown manual-review rate" in text
    assert "safety trade-off" in text
    assert "0.976562" in text


def test_pretrained_safe_hierarchical_summary_mentions_final_comparison():
    text = read_text(
        "docs/supervisor_updates/pretrained_safe_hierarchical_policy_tuning_summary_v1.md"
    )

    assert "pretrained transfer-learning model" in text
    assert "selected safe pretrained policy" in text
    assert "final model/policy comparison" in text