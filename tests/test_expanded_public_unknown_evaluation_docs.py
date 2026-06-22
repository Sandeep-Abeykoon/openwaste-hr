from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_expanded_public_unknown_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "expanded_public_unknown_evaluation_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "expanded_public_unknown_evaluation_summary_v1.md"
    ).exists()


def test_expanded_public_unknown_configs_exist():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "expanded_public_local_unknown_evaluation.yaml"
    ).exists()

    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "expanded_public_public_unknown_evaluation.yaml"
    ).exists()


def test_expanded_public_unknown_doc_mentions_baseline_c_and_sources():
    text = read_text("docs/methodology/expanded_public_unknown_evaluation_v1.md")

    assert "Baseline C" in text
    assert "pretrained expanded public dataset model" in text
    assert "local unknown dataset" in text
    assert "public unknown/future-class" in text
    assert "Textile Trash" in text


def test_expanded_public_local_unknown_config_uses_expanded_model():
    text = read_text("ml/configs/expanded_public_local_unknown_evaluation.yaml")

    assert "expanded_public_pretrained_v1_best.pt" in text
    assert "expanded_public_pretrained_v1_class_mapping.json" in text
    assert "local_unknown_manifest_v1.csv" in text
    assert "expanded_public_local_unknown" in text


def test_expanded_public_public_unknown_config_uses_public_unknown_manifest():
    text = read_text("ml/configs/expanded_public_public_unknown_evaluation.yaml")

    assert "expanded_public_pretrained_v1_best.pt" in text
    assert "expanded_public_pretrained_v1_class_mapping.json" in text
    assert "expanded_public_unknown_test_v1.csv" in text
    assert "expanded_public_public_unknown" in text


def test_expanded_public_unknown_configs_are_valid_yaml():
    local_config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "ml"
            / "configs"
            / "expanded_public_local_unknown_evaluation.yaml"
        ).read_text(encoding="utf-8")
    )

    public_config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "ml"
            / "configs"
            / "expanded_public_public_unknown_evaluation.yaml"
        ).read_text(encoding="utf-8")
    )

    assert local_config is not None
    assert public_config is not None