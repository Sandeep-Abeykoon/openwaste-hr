from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_pretrained_evaluation_config_exists():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "evaluate_pretrained_trashnet.yaml"
    ).exists()


def test_pretrained_evaluation_config_mentions_pretrained_checkpoint():
    text = read_text("ml/configs/evaluate_pretrained_trashnet.yaml")

    assert "pretrained_trashnet_v1" in text
    assert "pretrained_trashnet_v1_best.pt" in text


def test_pretrained_evaluation_methodology_doc_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "pretrained_baseline_evaluation_v1.md"
    ).exists()


def test_pretrained_evaluation_doc_mentions_baseline_comparison():
    text = read_text("docs/methodology/pretrained_baseline_evaluation_v1.md")

    assert "Baseline A" in text
    assert "Baseline B" in text
    assert "scratch-trained" in text
    assert "pretrained transfer-learning" in text
    assert "0.888000" in text


def test_pretrained_evaluation_summary_mentions_next_stages():
    text = read_text(
        "docs/supervisor_updates/pretrained_baseline_evaluation_summary_v1.md"
    )

    assert "confidence reject baseline" in text
    assert "local unknown evaluation" in text
    assert "hierarchical decision policy" in text
    assert "safe hierarchical tuning" in text