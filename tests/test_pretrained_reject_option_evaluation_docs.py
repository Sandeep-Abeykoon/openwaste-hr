from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_pretrained_reject_configs_exist():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "confidence_reject_pretrained_trashnet.yaml"
    ).exists()

    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "open_set_score_pretrained_trashnet.yaml"
    ).exists()


def test_pretrained_confidence_reject_config_points_to_pretrained_checkpoint():
    text = read_text("ml/configs/confidence_reject_pretrained_trashnet.yaml")

    assert "pretrained_trashnet_v1" in text
    assert "pretrained_trashnet_v1_best.pt" in text
    assert "pretrained_confidence_reject" in text


def test_pretrained_open_set_config_points_to_pretrained_checkpoint():
    text = read_text("ml/configs/open_set_score_pretrained_trashnet.yaml")

    assert "pretrained_trashnet_v1" in text
    assert "pretrained_trashnet_v1_best.pt" in text
    assert "pretrained_open_set_score" in text


def test_pretrained_reject_methodology_doc_exists_and_mentions_methods():
    text = read_text("docs/methodology/pretrained_reject_option_evaluation_v1.md")

    assert "confidence threshold" in text
    assert "max-logit" in text
    assert "energy" in text
    assert "selective accuracy" in text
    assert "Baseline B" in text


def test_pretrained_reject_summary_mentions_next_stage():
    text = read_text(
        "docs/supervisor_updates/pretrained_reject_option_evaluation_summary_v1.md"
    )

    assert "pretrained transfer-learning checkpoint" in text
    assert "local unknown dataset" in text
    assert "false acceptance" in text