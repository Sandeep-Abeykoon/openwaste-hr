from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.inference_schema import (
    InferenceApiResponse,
    InferenceRequest,
)
from backend.app.services import inference_service

router = APIRouter(
    prefix="/api/inference",
    tags=["OpenWaste-HR Inference"],
)


@router.post(
    "/predict",
    response_model=InferenceApiResponse,
)
def predict_image(request: InferenceRequest) -> dict:
    """
    Run OpenWaste-HR inference on one image.
    """
    response = inference_service.predict_from_request(
        image_path=request.image_path,
        sample_id=request.sample_id,
        request_id=request.request_id,
    )

    if response.get("status") == "error":
        raise HTTPException(
            status_code=400,
            detail=response,
        )

    return response