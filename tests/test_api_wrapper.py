import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.inference.api_wrapper import (  # noqa: E402
    build_error_response,
    build_success_response,
    normalize_extension,
    run_api_wrapper_prediction,
    validate_api_request,
)


def create_config(tmp_path: Path) -> Path:
    config = {
        "project": {
            "name": "OpenWaste-HR",
            "api_stage": "prototype_api_wrapper",
            "version": "v1",
        },
        "inference": {
            "single_image_config_path": "ml/configs/inference_pipeline.yaml",
        },
        "api": {
            "allowed_image_extensions": [".jpg", ".jpeg", ".png"],
            "max_image_path_length": 500,
        },
        "outputs": {
            "output_metrics_dir": str(tmp_path / "outputs"),
            "output_response_json": "prototype_api_response_v1.json",
            "output_response_markdown": "prototype_api_response_v1.md",
        },
    }

    config_path = tmp_path / "api_wrapper.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    return config_path


def create_inference_result() -> dict:
    return {
        "sample_id": "local_000001",
        "image_path": "ml/data/local_unknown/images/local_000001.jpg",
        "device": "cpu",
        "pred_label": "plastic",
        "max_softmax_confidence": 0.962933,
        "top_coarse_label": "recyclable",
        "top_coarse_confidence": 0.999999,
        "second_coarse_confidence": 0.000001,
        "coarse_margin": 0.999998,
        "hierarchical_decision_type": "fine_label",
        "hierarchical_final_label": "plastic",
        "hierarchical_final_confidence": 0.962933,
        "hierarchical_decision_reason": "fine_confidence_high",
        "class_probabilities": {
            "paper_cardboard": 0.002948,
            "plastic": 0.962933,
            "glass": 0.026874,
            "metal": 0.007244,
            "residual": 0.0,
        },
        "policy": {
            "fine_confidence_threshold": 0.9,
            "coarse_confidence_threshold": 0.8,
            "coarse_margin_threshold": 0.65,
            "minimum_confidence_for_coarse": 0.65,
        },
    }


def test_normalize_extension():
    assert normalize_extension("jpg") == ".jpg"
    assert normalize_extension(".PNG") == ".png"


def test_validate_api_request_success():
    request = {
        "image_path": "ml/data/local_unknown/images/local_000001.jpg",
        "sample_id": "local_000001",
    }

    normalized = validate_api_request(
        request_payload=request,
        allowed_image_extensions=[".jpg", ".png"],
        max_image_path_length=500,
    )

    assert normalized["image_path"] == "ml/data/local_unknown/images/local_000001.jpg"
    assert normalized["sample_id"] == "local_000001"


def test_validate_api_request_missing_image_path():
    with pytest.raises(ValueError, match="image_path is required"):
        validate_api_request(
            request_payload={},
            allowed_image_extensions=[".jpg"],
            max_image_path_length=500,
        )


def test_validate_api_request_rejects_bad_extension():
    with pytest.raises(ValueError, match="Unsupported image extension"):
        validate_api_request(
            request_payload={"image_path": "file.txt"},
            allowed_image_extensions=[".jpg"],
            max_image_path_length=500,
        )


def test_build_success_response():
    response = build_success_response(
        inference_result=create_inference_result(),
        request_id="req_001",
    )

    assert response["status"] == "success"
    assert response["request_id"] == "req_001"
    assert response["decision"]["decision_type"] == "fine_label"
    assert response["decision"]["final_label"] == "plastic"


def test_build_error_response():
    response = build_error_response(
        error=ValueError("bad request"),
        request_id="req_002",
    )

    assert response["status"] == "error"
    assert response["error"]["type"] == "ValueError"
    assert response["error"]["message"] == "bad request"


def test_run_api_wrapper_prediction_with_mocked_inference(tmp_path, monkeypatch):
    config_path = create_config(tmp_path)

    def fake_single_image_inference(*args, **kwargs):
        return create_inference_result()

    monkeypatch.setattr(
        "openwaste_hr.inference.api_wrapper.run_single_image_inference",
        fake_single_image_inference,
    )

    response = run_api_wrapper_prediction(
        config_path=config_path,
        project_root=PROJECT_ROOT,
        request_payload={
            "image_path": "ml/data/local_unknown/images/local_000001.jpg",
            "sample_id": "local_000001",
        },
        request_id="req_test",
    )

    assert response["status"] == "success"
    assert response["request_id"] == "req_test"
    assert response["decision"]["final_label"] == "plastic"

    assert (tmp_path / "outputs" / "prototype_api_response_v1.json").exists()
    assert (tmp_path / "outputs" / "prototype_api_response_v1.md").exists()