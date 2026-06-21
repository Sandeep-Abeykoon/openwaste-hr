from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_pretrained_smoke_config_exists():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "train_pretrained_trashnet_smoke.yaml"
    ).exists()


def test_pretrained_smoke_config_uses_pretrained():
    text = read_text("ml/configs/train_pretrained_trashnet_smoke.yaml").lower()

    assert "pretrained: true" in text
    assert "pretrained: false" not in text


def test_pretrained_smoke_config_uses_separate_output_name():
    text = read_text("ml/configs/train_pretrained_trashnet_smoke.yaml")

    assert "pretrained_trashnet_smoke_v1" in text


def test_pretrained_smoke_methodology_doc_exists_and_mentions_smoke_test():
    text = read_text("docs/methodology/pretrained_training_smoke_test_v1.md")

    assert "smoke test" in text.lower()
    assert "Baseline B" in text
    assert "pretrained transfer-learning" in text
    assert "not the final pretrained comparison result" in text


def test_pretrained_smoke_supervisor_summary_exists_and_mentions_full_training():
    text = read_text(
        "docs/supervisor_updates/pretrained_training_smoke_test_summary_v1.md"
    )

    assert "pretrained: true" in text
    assert "full pretrained training" in text
    assert "not used as the final pretrained model comparison" in text