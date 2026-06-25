import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))

from backend.app.main import create_app  # noqa: E402
from backend.app.services import inference_service  # noqa: E402


def create_fake_success_response() -> dict:
    return {
        "status": "success",
        "request_id": "req_test",
        "request": {
            "sample_id": "local_000001",
            "image_path": "ml/data/local_unknown/images/local_000001.jpg",
        },
        "prediction": {
            "pred_label": "plastic",
            "max_softmax_confidence": 0.962933,
            "top_coarse_label": "recyclable",
            "top_coarse_confidence": 0.999999,
            "coarse_margin": 0.999999,
        },
        "decision": {
            "decision_type": "fine_label",
            "final_label": "plastic",
            "final_confidence": 0.962933,
            "reason": "fine_confidence_high",
        },
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
        "metadata": {
            "device": "cpu",
            "pipeline_version": "prototype_api_wrapper_v1",
        },
    }


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "openwaste-hr-backend"


def test_predict_endpoint_success(monkeypatch):
    app = create_app()
    client = TestClient(app)

    def fake_predict_from_request(*args, **kwargs):
        return create_fake_success_response()

    monkeypatch.setattr(
        inference_service,
        "predict_from_request",
        fake_predict_from_request,
    )

    response = client.post(
        "/api/inference/predict",
        json={
            "image_path": "ml/data/local_unknown/images/local_000001.jpg",
            "sample_id": "local_000001",
            "request_id": "req_test",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["decision"]["decision_type"] == "fine_label"
    assert data["decision"]["final_label"] == "plastic"


def test_predict_endpoint_error(monkeypatch):
    app = create_app()
    client = TestClient(app)

    def fake_predict_from_request(*args, **kwargs):
        return {
            "status": "error",
            "request_id": "req_error",
            "error": {
                "type": "ValueError",
                "message": "bad request",
            },
            "metadata": {
                "pipeline_version": "prototype_api_wrapper_v1",
            },
        }

    monkeypatch.setattr(
        inference_service,
        "predict_from_request",
        fake_predict_from_request,
    )

    response = client.post(
        "/api/inference/predict",
        json={
            "image_path": "bad.txt",
            "sample_id": "bad",
            "request_id": "req_error",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["status"] == "error"


def test_validate_project_relative_image_path_rejects_escape():
    with pytest.raises(ValueError, match="inside the project directory"):
        inference_service.validate_project_relative_image_path(
            project_root=PROJECT_ROOT,
            image_path="../outside.jpg",
        )