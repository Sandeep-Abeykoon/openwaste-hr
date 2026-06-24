from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_pretrained_training_config_exists():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "train_pretrained_trashnet.yaml"
    ).exists()


def test_pretrained_training_config_enables_pretrained():
    text = read_text("ml/configs/train_pretrained_trashnet.yaml").lower()

    assert "pretrained: true" in text
    assert "pretrained: false" not in text


def test_pretrained_training_config_uses_separate_output_name():
    text = read_text("ml/configs/train_pretrained_trashnet.yaml")

    assert "pretrained_trashnet_v1" in text
