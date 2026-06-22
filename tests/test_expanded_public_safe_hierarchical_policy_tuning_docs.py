from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_expanded_public_safe_hierarchical_policy_tuning_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "expanded_public_safe_hierarchical_policy_tuning_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "expanded_public_safe_hierarchical_policy_tuning_summary_v1.md"
    ).exists()


def test_expanded_public_safe_hierarchical_policy_config_exists():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "expanded_public_safe_hierarchical_policy_tuning.yaml"
    ).exists()


def test_expanded_public_safe_hierarchical_doc_mentions_policy_outputs():
    text = read_text(
        "docs/methodology/expanded_public_safe_hierarchical_policy_tuning_v1.md"
    )

    assert "Baseline C" in text
    assert "pretrained expanded public dataset model" in text
    assert "fine label" in text
    assert "coarse fallback" in text
    assert "manual review" in text


def test_expanded_public_safe_hierarchical_config_uses_expanded_model_and_data():
    text = read_text("ml/configs/expanded_public_safe_hierarchical_policy_tuning.yaml")

    assert "expanded_public_pretrained_v1_best.pt" in text
    assert "expanded_public_pretrained_v1_class_mapping.json" in text
    assert "expanded_public_known_test_v1.csv" in text
    assert "local_unknown_manifest_v1.csv" in text
    assert "expanded_public_safe_hierarchical" in text


def test_expanded_public_safe_hierarchical_doc_mentions_previous_findings():
    text = read_text(
        "docs/methodology/expanded_public_safe_hierarchical_policy_tuning_v1.md"
    )

    assert "closed-set" in text
    assert "reject-option" in text
    assert "unknown rejection" in text
    assert "energy score" in text
    assert "confidence threshold" in text


def test_expanded_public_safe_hierarchical_config_is_valid_yaml():
    config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "ml"
            / "configs"
            / "expanded_public_safe_hierarchical_policy_tuning.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config is not None