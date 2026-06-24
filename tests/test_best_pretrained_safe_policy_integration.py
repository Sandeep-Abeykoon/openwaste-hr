from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CORE_CONFIGS = [
    "ml/configs/inference_pipeline.yaml",
    "ml/configs/batch_inference_pipeline.yaml",
    "ml/configs/prototype_api_wrapper.yaml",
]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_core_configs_use_pretrained_best_checkpoint():
    for relative_path in CORE_CONFIGS:
        text = read_text(relative_path)

        assert "pretrained_trashnet_v1" in text
        assert "pretrained_trashnet_v1_best.pt" in text


def test_core_configs_use_best_safe_policy_thresholds():
    for relative_path in CORE_CONFIGS:
        text = read_text(relative_path)

        assert "fine_confidence_threshold: 0.995" in text
        assert "coarse_confidence_threshold: 0.8" in text
        assert "coarse_margin_threshold: 0.15" in text
        assert "minimum_confidence_for_coarse: 0.9" in text


def test_frontend_mentions_best_policy():
    text = read_text("frontend/index.html")

    assert "OpenWaste-HR Best Prototype" in text
    assert "pretrained safe hierarchical policy" in text
