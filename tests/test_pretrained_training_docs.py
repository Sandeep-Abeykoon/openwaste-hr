from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_pretrained_training_methodology_doc_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "pretrained_training_v1.md"
    ).exists()


def test_pretrained_training_supervisor_doc_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "pretrained_training_summary_v1.md"
    ).exists()


def test_pretrained_training_doc_mentions_baselines_and_config():
    text = read_text("docs/methodology/pretrained_training_v1.md")

    assert "Baseline A" in text
    assert "Baseline B" in text
    assert "train_pretrained_trashnet.yaml" in text
    assert "pretrained: true" in text
    assert "pretrained_trashnet_v1" in text


def test_pretrained_training_doc_mentions_future_evaluation():
    text = read_text("docs/methodology/pretrained_training_v1.md")

    assert "closed-set metrics" in text
    assert "local unknown evaluation" in text
    assert "hierarchical policy evaluation" in text
    assert "safe policy tuning" in text


def test_pretrained_training_summary_mentions_project_contribution():
    text = read_text("docs/supervisor_updates/pretrained_training_summary_v1.md")

    assert "pretrained transfer-learning model" in text
    assert "scratch-trained baseline" in text
    assert "hierarchical open-set waste classification" in text
    assert "manual-review" in text