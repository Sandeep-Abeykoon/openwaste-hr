from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ml.api.manual_review_queue import (
    delete_manual_review,
    get_intelligence_candidates_path,
    list_manual_reviews,
    queue_manual_review,
    resolve_review_image_path,
    save_intelligence_candidates,
    update_review_decision,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "ml" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from infer_with_fusion_gate_v2_policy import (  # noqa: E402
    FUSION_FEATURES,
    build_fusion_feature_vector,
    compute_mahalanobis_knownness,
    compute_model_scores,
    load_classifier,
    make_user_message,
    preprocess_image,
    read_yaml,
    run_classifier_with_embedding,
)
from train_image_classifier import build_transforms  # noqa: E402


def should_force_cpu() -> bool:
    value = os.getenv("OPENWASTE_FORCE_CPU", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_cuda_runtime_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in [
            "cuda error",
            "cuda runtime",
            "cudnn",
            "device-side assert",
            "cuda kernel",
        ]
    )


class PredictionService:
    def __init__(self, preferred_device: str | None = None) -> None:
        self.training_config_path = REPO_ROOT / "ml" / "configs" / "train_stage_04_add_trashbox_clean.yaml"
        self.policy_config_path = REPO_ROOT / "ml" / "configs" / "final_decision_policy_v2_fusion_gate.yaml"

        self.training_config = read_yaml(self.training_config_path)
        self.policy_config = read_yaml(self.policy_config_path)

        self.checkpoint_path = REPO_ROOT / self.policy_config["base_model"]["checkpoint_path"]
        self.mahalanobis_model_path = REPO_ROOT / self.policy_config["mahalanobis"]["model_path"]
        self.fusion_gate_model_path = REPO_ROOT / self.policy_config["fusion_gate"]["model_path"]

        self.threshold = float(self.policy_config["fusion_gate"]["threshold"])
        self.temperature = float(self.policy_config["temperature_scaling"]["temperature"])

        self.device = self.resolve_device(preferred_device)

        self.model, self.class_names = load_classifier(
            training_config=self.training_config,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        self.transform = build_transforms(
            REPO_ROOT / self.training_config["preprocessing"]["config_path"],
            train=False,
        )

        self.mahalanobis_model = joblib.load(self.mahalanobis_model_path)
        self.fusion_gate = joblib.load(self.fusion_gate_model_path)

    @staticmethod
    def resolve_device(preferred_device: str | None = None) -> torch.device:
        requested_device = str(
            preferred_device
            or os.getenv("OPENWASTE_INFERENCE_DEVICE", "")
        ).strip().lower()

        if should_force_cpu() or requested_device == "cpu":
            return torch.device("cpu")

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "OPENWASTE_INFERENCE_DEVICE=cuda was requested, but CUDA is not available."
                )
            return torch.device("cuda")

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, image_path: Path) -> dict[str, Any]:
        image_tensor = preprocess_image(
            image_path=image_path,
            transform=self.transform,
            device=self.device,
        )

        logits, embedding, embedding_layer = run_classifier_with_embedding(
            model=self.model,
            image_tensor=image_tensor,
        )

        model_scores = compute_model_scores(
            logits=logits,
            class_names=self.class_names,
            temperature=self.temperature,
        )

        mahalanobis_scores = compute_mahalanobis_knownness(
            embedding=embedding,
            mahalanobis_model=self.mahalanobis_model,
            class_names=self.class_names,
        )

        combined_scores = {
            **model_scores,
            **mahalanobis_scores,
        }

        fusion_feature_vector = build_fusion_feature_vector(combined_scores)
        fusion_knownness_score = float(self.fusion_gate.predict_proba(fusion_feature_vector)[0, 1])

        accepted = bool(fusion_knownness_score >= self.threshold)

        decision_type = "known_fine_label" if accepted else "unknown_manual_review"
        coarse_label = "recyclable" if accepted else "manual_review_required"

        result = {
            "policy_version": self.policy_config["policy_version"],
            "image_path": str(image_path),
            "device": str(self.device),
            "temperature": self.temperature,
            "embedding_layer": embedding_layer,
            "embedding_dimension": int(embedding.shape[0]),
            "known_classes": self.class_names,
            "prediction": {
                "internal_top1_prediction": combined_scores["pred_label"],
                "pred_index": combined_scores["pred_index"],
                "raw_confidence": combined_scores["confidence"],
                "temperature_scaled_confidence": combined_scores["temperature_scaled_confidence"],
                "max_logit": combined_scores["max_logit"],
                "energy": combined_scores["energy"],
                "softmax_margin": combined_scores["softmax_margin"],
                "softmax_entropy": combined_scores["softmax_entropy"],
                "class_probabilities": {
                    class_name: combined_scores[f"prob_{class_name}"]
                    for class_name in self.class_names
                },
            },
            "mahalanobis": {
                "mahalanobis_min_distance": combined_scores["mahalanobis_min_distance"],
                "mahalanobis_knownness": combined_scores["mahalanobis_knownness"],
                "mahalanobis_nearest_class": combined_scores["mahalanobis_nearest_class"],
            },
            "fusion_gate": {
                "feature_names": FUSION_FEATURES,
                "knownness_score": fusion_knownness_score,
                "threshold": self.threshold,
                "accepted_as_known": accepted,
                "decision_type": decision_type,
            },
            "final_decision": {
                "accepted_as_known": accepted,
                "decision_type": decision_type,
                "user_visible_label": combined_scores["pred_label"] if accepted else "manual_review_required",
                "coarse_label": coarse_label,
                "show_internal_prediction_to_user": accepted,
                "internal_top1_prediction_logged": True,
                "user_message": make_user_message(
                    accepted=accepted,
                    pred_label=combined_scores["pred_label"],
                    coarse_label="recyclable",
                ),
            },
        }

        return result


class ManualReviewDecisionRequest(BaseModel):
    selected_label: str | None = Field(default=None)
    custom_label: str | None = Field(default=None)
    review_notes: str | None = Field(default=None)
    promote_to_intelligence: bool = Field(default=False)


class ManualReviewQueueRequest(BaseModel):
    uploaded_filename: str | None = Field(default=None)
    stored_image_path: str | None = Field(default=None)
    result: dict[str, Any] = Field(default_factory=dict)


def build_manual_review_item_response(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **item,
        "image_url": f"/api/manual-review/{item['review_id']}/image",
    }


def build_manual_review_queue_response() -> dict[str, Any]:
    queue_payload = list_manual_reviews(REPO_ROOT)
    return {
        "items": [
            build_manual_review_item_response(item)
            for item in queue_payload["items"]
        ],
        "summary": queue_payload["summary"],
    }


def rebuild_prediction_service(preferred_device: str | None = None) -> PredictionService:
    return PredictionService(preferred_device=preferred_device)


def predict_with_fallback(image_path: Path) -> dict[str, Any]:
    global service

    try:
        return service.predict(image_path)
    except Exception as error:
        if service.device.type != "cuda" or not is_cuda_runtime_error(error):
            raise

        print(
            "CUDA inference failed. Falling back to CPU for live inference. "
            f"Original error: {error}",
            flush=True,
        )

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        service = rebuild_prediction_service(preferred_device="cpu")
        return service.predict(image_path)


app = FastAPI(
    title="OpenWaste-HR Prediction API",
    version="1.0.0",
    description="Prediction API for OpenWaste-HR Fusion Gate v2 waste classification.",
)

default_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

extra_allowed_origins = [
    origin.strip()
    for origin in os.getenv("OPENWASTE_UI_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[*default_allowed_origins, *extra_allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = rebuild_prediction_service()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "OpenWaste-HR Prediction API is running.",
        "policy_version": service.policy_config["policy_version"],
        "device": str(service.device),
        "known_classes": service.class_names,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "policy_version": service.policy_config["policy_version"],
        "device": str(service.device),
        "known_classes": service.class_names,
        "threshold": service.threshold,
        "temperature": service.temperature,
    }


@app.get("/api/manual-review")
def get_manual_review_queue() -> dict[str, Any]:
    return build_manual_review_queue_response()


@app.post("/api/manual-review/queue")
def queue_current_result_for_manual_review(
    payload: ManualReviewQueueRequest,
) -> dict[str, Any]:
    uploaded_filename = str(payload.uploaded_filename or "").strip()
    stored_image_path = str(payload.stored_image_path or "").strip()

    if uploaded_filename == "":
        raise HTTPException(
            status_code=400,
            detail="uploaded_filename is required to queue a manual review item.",
        )

    if stored_image_path == "":
        raise HTTPException(
            status_code=400,
            detail="stored_image_path is required to queue a manual review item.",
        )

    if not payload.result:
        raise HTTPException(
            status_code=400,
            detail="result payload is required to queue a manual review item.",
        )

    image_path = Path(stored_image_path)
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Stored inference image not found: {stored_image_path}",
        )

    response_payload = {
        "uploaded_filename": uploaded_filename,
        "stored_image_path": stored_image_path,
        "result": payload.result,
    }

    review_entry, was_created = queue_manual_review(
        repo_root=REPO_ROOT,
        review_id=uuid.uuid4().hex,
        response_payload=response_payload,
    )
    queue_payload = build_manual_review_queue_response()

    return {
        "item": build_manual_review_item_response(review_entry),
        "summary": queue_payload["summary"],
        "queue_status": "queued" if was_created else "existing",
    }


@app.get("/api/manual-review/intelligence/export")
def export_intelligence_candidates() -> FileResponse:
    queue_payload = list_manual_reviews(REPO_ROOT)
    save_intelligence_candidates(REPO_ROOT, queue_payload["items"])
    export_path = get_intelligence_candidates_path(REPO_ROOT)

    return FileResponse(
        export_path,
        media_type="application/json",
        filename="intelligence_candidates.json",
    )


@app.get("/api/manual-review/{review_id}/image")
def get_manual_review_image(review_id: str) -> FileResponse:
    try:
        image_path = resolve_review_image_path(REPO_ROOT, review_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return FileResponse(image_path)


@app.post("/api/manual-review/{review_id}/decision")
def submit_manual_review_decision(
    review_id: str,
    payload: ManualReviewDecisionRequest,
) -> dict[str, Any]:
    try:
        updated_item = update_review_decision(
            repo_root=REPO_ROOT,
            review_id=review_id,
            selected_label=payload.selected_label,
            custom_label=payload.custom_label,
            review_notes=payload.review_notes,
            promote_to_intelligence=payload.promote_to_intelligence,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    queue_payload = build_manual_review_queue_response()
    return {
        "item": build_manual_review_item_response(updated_item),
        "summary": queue_payload["summary"],
    }


@app.delete("/api/manual-review/{review_id}")
def remove_manual_review_item(review_id: str) -> dict[str, Any]:
    try:
        deleted_item, deleted_image_from_system = delete_manual_review(
            repo_root=REPO_ROOT,
            review_id=review_id,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    queue_payload = build_manual_review_queue_response()
    return {
        "deleted_review_id": deleted_item["review_id"],
        "deleted_image_from_system": deleted_image_from_system,
        "summary": queue_payload["summary"],
    }


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    uploads_dir = REPO_ROOT / "ml" / "outputs" / "api_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "uploaded_image.jpg").suffix
    if suffix.strip() == "":
        suffix = ".jpg"

    safe_name = f"{uuid.uuid4().hex}{suffix}"
    image_path = uploads_dir / safe_name

    try:
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_with_fallback(image_path)

        response = {
            "uploaded_filename": file.filename,
            "stored_image_path": str(image_path),
            "result": result,
            "manual_review_entry": None,
            "manual_review_entry_status": None,
        }

        if not result["final_decision"]["accepted_as_known"]:
            review_entry, was_created = queue_manual_review(
                repo_root=REPO_ROOT,
                review_id=uuid.uuid4().hex,
                response_payload=response,
            )
            response["manual_review_entry"] = build_manual_review_item_response(
                review_entry
            )
            response["manual_review_entry_status"] = (
                "queued" if was_created else "existing"
            )

        return response

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error
