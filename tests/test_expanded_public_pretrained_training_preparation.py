from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_expanded_public_training_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "expanded_public_pretrained_training_preparation_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "expanded_public_pretrained_training_preparation_summary_v1.md"
    ).exists()


def test_expanded_public_training_configs_exist():
    assert (
        PROJECT_ROOT / "ml" / "configs" / "train_expanded_public_pretrained.yaml"
    ).exists()

    assert (
        PROJECT_ROOT / "ml" / "configs" / "train_expanded_public_pretrained_smoke.yaml"
    ).exists()


def test_expanded_public_docs_mention_baseline_c():
    text = read_text(
        "docs/methodology/expanded_public_pretrained_training_preparation_v1.md"
    )

    assert "Baseline C" in text
    assert "pretrained expanded public dataset model" in text
    assert "TrashNet-style and RealWaste" in text
    assert "4869" in text
    assert "1042" in text
    assert "1050" in text


def test_expanded_public_full_config_uses_expanded_manifests():
    text = read_text("ml/configs/train_expanded_public_pretrained.yaml")

    assert "expanded_public_known_train_v1.csv" in text
    assert "expanded_public_known_val_v1.csv" in text
    assert "expanded_public_known_test_v1.csv" in text
    assert "expanded_public_pretrained_v1" in text
    assert "pretrained: true" in text


def test_expanded_public_smoke_config_uses_smoke_outputs_and_one_epoch():
    text = read_text("ml/configs/train_expanded_public_pretrained_smoke.yaml")

    assert "expanded_public_known_train_v1.csv" in text
    assert "expanded_public_known_val_v1.csv" in text
    assert "expanded_public_known_test_v1.csv" in text
    assert "expanded_public_pretrained_smoke_v1" in text
    assert "epochs: 1" in text
    assert "patience: 1" in text


def test_expanded_public_configs_are_valid_yaml():
    full_config = yaml.safe_load(
        (PROJECT_ROOT / "ml" / "configs" / "train_expanded_public_pretrained.yaml")
        .read_text(encoding="utf-8")
    )
    smoke_config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "ml"
            / "configs"
            / "train_expanded_public_pretrained_smoke.yaml"
        ).read_text(encoding="utf-8")
    )

    assert full_config is not None
    assert smoke_config is not None