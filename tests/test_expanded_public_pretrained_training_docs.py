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
        / "expanded_public_pretrained_training_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "expanded_public_pretrained_training_summary_v1.md"
    ).exists()


def test_expanded_public_training_doc_mentions_baseline_c():
    text = read_text("docs/methodology/expanded_public_pretrained_training_v1.md")

    assert "Baseline C" in text
    assert "pretrained expanded public dataset model" in text
    assert "TrashNet-style dataset" in text
    assert "RealWaste" in text


def test_expanded_public_training_doc_mentions_splits_and_classes():
    text = read_text("docs/methodology/expanded_public_pretrained_training_v1.md")

    assert "4869" in text
    assert "1042" in text
    assert "1050" in text
    assert "paper_cardboard" in text
    assert "organic" in text
    assert "residual" in text


def test_expanded_public_training_doc_mentions_unknown_exclusion():
    text = read_text("docs/methodology/expanded_public_pretrained_training_v1.md")

    assert "unknown" in text
    assert "not used as a training class" in text
    assert "Textile Trash" in text
    assert "open-set design" in text


def test_expanded_public_training_config_uses_correct_manifests():
    text = read_text("ml/configs/train_expanded_public_pretrained.yaml")

    assert "expanded_public_known_train_v1.csv" in text
    assert "expanded_public_known_val_v1.csv" in text
    assert "expanded_public_known_test_v1.csv" in text
    assert "expanded_public_pretrained_v1" in text
    assert "pretrained: true" in text


def test_expanded_public_training_config_is_valid_yaml():
    config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "ml"
            / "configs"
            / "train_expanded_public_pretrained.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config is not None