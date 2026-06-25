from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.inference_routes import router as inference_router
from backend.app.schemas.inference_schema import HealthResponse


APP_VERSION = "backend_inference_endpoint_v1"


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenWaste-HR Backend",
        description="Prototype backend for hierarchical open-set waste classification.",
        version=APP_VERSION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(inference_router)

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "openwaste-hr-backend",
            "version": APP_VERSION,
        }

    return app


app = create_app()