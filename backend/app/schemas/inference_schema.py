from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    image_path: str = Field(
        ...,
        description="Project-relative path to the image.",
        examples=["ml/data/local_unknown/images/local_000001.jpg"],
    )
    sample_id: str | None = Field(
        default=None,
        description="Optional sample ID. If omitted, the image filename stem is used.",
        examples=["local_000001"],
    )
    request_id: str | None = Field(
        default=None,
        description="Optional request ID for tracing.",
        examples=["demo_request_001"],
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class InferenceApiResponse(BaseModel):
    status: str
    request_id: str
    request: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    class_probabilities: dict[str, float] | None = None
    policy: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None
    error: dict[str, Any] | None = None