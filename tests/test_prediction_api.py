from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from ml.api import predict_api
from ml.api.manual_review_queue import list_manual_reviews

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeDevice:
    def __init__(self, device_type: str = "cpu") -> None:
        self.type = device_type

    def __str__(self) -> str:
        return self.type


class FakePredictionService:
    def __init__(self, *, accepted_as_known: bool) -> None:
        self.accepted_as_known = accepted_as_known
        self.policy_config = {
            "policy_version": "test_policy_v1",
        }
        self.device = FakeDevice()
        self.class_names = [
            "cardboard",
            "glass",
            "metal",
            "paper",
            "plastic",
        ]
        self.threshold = 0.63
        self.temperature = 0.94

    def predict(self, image_path: Path) -> dict:
        knownness_score = 0.81 if self.accepted_as_known else 0.21
        decision_type = (
            "known_fine_label"
            if self.accepted_as_known
            else "unknown_manual_review"
        )
        user_visible_label = (
            "paper"
            if self.accepted_as_known
            else "manual_review_required"
        )

        return {
            "policy_version": self.policy_config["policy_version"],
            "image_path": str(image_path),
            "device": str(self.device),
            "temperature": self.temperature,
            "embedding_layer": "classifier.3",
            "embedding_dimension": 1280,
            "known_classes": self.class_names,
            "prediction": {
                "internal_top1_prediction": "paper",
            },
            "mahalanobis": {
                "mahalanobis_min_distance": 1.23,
                "mahalanobis_knownness": 0.45,
                "mahalanobis_nearest_class": "paper",
            },
            "fusion_gate": {
                "feature_names": ["confidence", "energy"],
                "knownness_score": knownness_score,
                "threshold": self.threshold,
                "accepted_as_known": self.accepted_as_known,
                "decision_type": decision_type,
            },
            "final_decision": {
                "accepted_as_known": self.accepted_as_known,
                "decision_type": decision_type,
                "user_visible_label": user_visible_label,
                "coarse_label": user_visible_label,
                "show_internal_prediction_to_user": self.accepted_as_known,
                "internal_top1_prediction_logged": True,
                "user_message": (
                    "Accepted known item."
                    if self.accepted_as_known
                    else "Please review this sample."
                ),
            },
        }


def create_client(
    monkeypatch,
    *,
    repo_root: Path,
    accepted_as_known: bool,
) -> TestClient:
    monkeypatch.setattr(predict_api, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        predict_api,
        "service",
        FakePredictionService(accepted_as_known=accepted_as_known),
    )
    return TestClient(predict_api.app)


def test_health_endpoint_uses_prediction_service_metadata(monkeypatch):
    temp_root = PROJECT_ROOT / "tmp_test_runs"
    temp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=temp_root) as temp_dir:
        client = create_client(
            monkeypatch,
            repo_root=Path(temp_dir),
            accepted_as_known=True,
        )

        response = client.get("/api/health")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["policy_version"] == "test_policy_v1"
        assert data["device"] == "cpu"
        assert data["threshold"] == 0.63


def test_predict_endpoint_rejects_non_image_upload(monkeypatch):
    temp_root = PROJECT_ROOT / "tmp_test_runs"
    temp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=temp_root) as temp_dir:
        client = create_client(
            monkeypatch,
            repo_root=Path(temp_dir),
            accepted_as_known=True,
        )

        response = client.post(
            "/api/predict",
            files={"file": ("sample.txt", b"not-an-image", "text/plain")},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Please upload a valid image file."


def test_predict_endpoint_returns_known_result_without_manual_review(monkeypatch):
    temp_root = PROJECT_ROOT / "tmp_test_runs"
    temp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=temp_root) as temp_dir:
        repo_root = Path(temp_dir)
        client = create_client(
            monkeypatch,
            repo_root=repo_root,
            accepted_as_known=True,
        )

        response = client.post(
            "/api/predict",
            files={"file": ("known-item.jpg", b"fake-image", "image/jpeg")},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["result"]["final_decision"]["accepted_as_known"] is True
        assert data["manual_review_entry"] is None
        assert data["manual_review_entry_status"] is None
        assert Path(data["stored_image_path"]).exists()
        assert list_manual_reviews(repo_root)["summary"]["total_count"] == 0


def test_predict_endpoint_queues_unknown_result_for_manual_review(monkeypatch):
    temp_root = PROJECT_ROOT / "tmp_test_runs"
    temp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=temp_root) as temp_dir:
        repo_root = Path(temp_dir)
        client = create_client(
            monkeypatch,
            repo_root=repo_root,
            accepted_as_known=False,
        )

        response = client.post(
            "/api/predict",
            files={"file": ("unknown-item.jpg", b"fake-image", "image/jpeg")},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["result"]["final_decision"]["accepted_as_known"] is False
        assert data["manual_review_entry_status"] == "queued"
        assert data["manual_review_entry"]["summary"]["decision_type"] == (
            "unknown_manual_review"
        )

        queue_response = client.get("/api/manual-review")
        assert queue_response.status_code == 200
        assert queue_response.json()["summary"]["pending_count"] == 1
