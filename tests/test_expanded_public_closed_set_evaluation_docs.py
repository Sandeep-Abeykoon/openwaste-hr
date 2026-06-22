from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_expanded_public_closed_set_eval_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "expanded_public_closed_set_evaluation_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "expanded_public_closed_set_evaluation_summary_v1.md"
    ).exists()


def test_expanded_public_closed_set_eval_config_exists():
    assert (
        PROJECT_ROOT / "ml" / "configs" / "evaluate_expanded_public_pretrained.yaml"
    ).exists()


def test_expanded_public_closed_set_eval_doc_mentions_baseline_c():
    text = read_text("docs/methodology/expanded_public_closed_set_evaluation_v1.md")

    assert "Baseline C" in text
    assert "pretrained expanded public dataset model" in text
    assert "TrashNet-style and RealWaste" in text
    assert "1050" in text


def test_expanded_public_closed_set_eval_doc_mentions_metrics_and_comparison():
    text = read_text("docs/methodology/expanded_public_closed_set_evaluation_v1.md")

    assert "accuracy" in text
    assert "balanced accuracy" in text
    assert "macro-F1" in text
    assert "weighted-F1" in text
    assert "0.8876" in text
    assert "0.8819" in text


def test_expanded_public_closed_set_eval_config_uses_expanded_paths():
    text = read_text("ml/configs/evaluate_expanded_public_pretrained.yaml")

    assert "expanded_public_known_test_v1.csv" in text
    assert "expanded_public_pretrained_v1_best.pt" in text
    assert "expanded_public_pretrained_v1_class_mapping.json" in text
    assert "expanded_public_pretrained" in text


def test_expanded_public_closed_set_eval_config_is_valid_yaml():
    config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "ml"
            / "configs"
            / "evaluate_expanded_public_pretrained.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config is not None