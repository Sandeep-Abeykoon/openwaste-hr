from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_pretrained_local_unknown_config_exists():
    assert (
        PROJECT_ROOT
        / "ml"
        / "configs"
        / "pretrained_local_unknown_evaluation.yaml"
    ).exists()


def test_pretrained_local_unknown_config_points_to_pretrained_assets():
    text = read_text("ml/configs/pretrained_local_unknown_evaluation.yaml")

    assert "pretrained_trashnet_v1" in text
    assert "pretrained_trashnet_v1_best.pt" in text
    assert "pretrained_confidence_reject" in text
    assert "pretrained_open_set_score" in text


def test_pretrained_local_unknown_config_uses_separate_outputs():
    text = read_text("ml/configs/pretrained_local_unknown_evaluation.yaml")

    assert "pretrained_local_unknown_predictions_v1.csv" in text
    assert "pretrained_local_unknown_evaluation_metrics_v1.json" in text


def test_pretrained_local_unknown_methodology_doc_mentions_comparison():
    text = read_text("docs/methodology/pretrained_local_unknown_evaluation_v1.md")

    assert "Baseline A" in text
    assert "Baseline B" in text
    assert "local unknown dataset" in text.lower()
    assert "unknown_false_acceptance_rate" in text
    assert "0.350000" in text


def test_pretrained_local_unknown_summary_mentions_next_stage():
    text = read_text(
        "docs/supervisor_updates/pretrained_local_unknown_evaluation_summary_v1.md"
    )

    assert "pretrained transfer-learning model" in text
    assert "unknown_rejection_rate" in text
    assert "hierarchical decision" in text
    assert "safe policy tuning" in text