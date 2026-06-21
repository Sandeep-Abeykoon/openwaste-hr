from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CORE_CONFIGS = [
    "ml/configs/inference_pipeline.yaml",
    "ml/configs/batch_inference_pipeline.yaml",
    "ml/configs/prototype_api_wrapper.yaml",
]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_best_policy_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "best_pretrained_safe_policy_integration_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "best_pretrained_safe_policy_integration_summary_v1.md"
    ).exists()


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


def test_methodology_doc_mentions_best_result():
    text = read_text("docs/methodology/best_pretrained_safe_policy_integration_v1.md")

    assert "Pretrained Safe Hierarchical Policy" in text
    assert "0.864583" in text
    assert "0.960843" in text
    assert "0.600000" in text


def test_supervisor_summary_mentions_updated_components():
    text = read_text(
        "docs/supervisor_updates/best_pretrained_safe_policy_integration_summary_v1.md"
    )

    assert "single-image inference" in text
    assert "batch inference" in text
    assert "API wrapper" in text
    assert "frontend demo" in text