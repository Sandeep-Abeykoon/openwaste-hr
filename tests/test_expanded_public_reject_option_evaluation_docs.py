from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_expanded_public_reject_option_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "expanded_public_reject_option_evaluation_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "expanded_public_reject_option_evaluation_summary_v1.md"
    ).exists()


def test_expanded_public_reject_option_configs_exist():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "confidence_reject_expanded_public_pretrained.yaml"
    ).exists()

    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "open_set_score_expanded_public_pretrained.yaml"
    ).exists()


def test_expanded_public_reject_option_doc_mentions_baseline_c_and_methods():
    text = read_text("docs/methodology/expanded_public_reject_option_evaluation_v1.md")

    assert "Baseline C" in text
    assert "pretrained expanded public dataset model" in text
    assert "confidence threshold" in text
    assert "max-logit" in text
    assert "energy score" in text


def test_confidence_reject_config_uses_expanded_public_paths():
    text = read_text("ml/configs/confidence_reject_expanded_public_pretrained.yaml")

    assert "expanded_public_known_val_v1.csv" in text
    assert "expanded_public_known_test_v1.csv" in text
    assert "expanded_public_pretrained_v1_best.pt" in text
    assert "expanded_public_pretrained_v1_class_mapping.json" in text
    assert "expanded_public_pretrained" in text


def test_open_set_score_config_uses_expanded_public_paths():
    text = read_text("ml/configs/open_set_score_expanded_public_pretrained.yaml")

    assert "expanded_public_known_val_v1.csv" in text
    assert "expanded_public_known_test_v1.csv" in text
    assert "expanded_public_pretrained_v1_best.pt" in text
    assert "expanded_public_pretrained_v1_class_mapping.json" in text
    assert "expanded_public_pretrained" in text


def test_expanded_public_reject_option_configs_are_valid_yaml():
    confidence_config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "ml"
            / "configs"
            / "confidence_reject_expanded_public_pretrained.yaml"
        ).read_text(encoding="utf-8")
    )

    open_set_config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "ml"
            / "configs"
            / "open_set_score_expanded_public_pretrained.yaml"
        ).read_text(encoding="utf-8")
    )

    assert confidence_config is not None
    assert open_set_config is not None