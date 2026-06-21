from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openwaste_hr.inference.api_wrapper import run_api_wrapper_prediction


def get_project_root() -> Path:
    """
    Resolve the OpenWaste-HR project root.

    Default structure:
        project_root/
            backend/app/services/inference_service.py
            ml/configs/prototype_api_wrapper.yaml
    """
    env_project_root = os.getenv("OPENWASTE_PROJECT_ROOT")

    if env_project_root:
        return Path(env_project_root).resolve()

    return Path(__file__).resolve().parents[3]


def get_default_api_config_path(project_root: Path) -> Path:
    return project_root / "ml" / "configs" / "prototype_api_wrapper.yaml"


def validate_project_relative_image_path(
    project_root: Path,
    image_path: str,
) -> str:
    """
    Validate that image_path is project-relative and does not escape the project root.
    """
    image_path = str(image_path).strip()

    if not image_path:
        raise ValueError("image_path is required.")

    candidate_path = (project_root / image_path).resolve()

    try:
        candidate_path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("image_path must stay inside the project directory.") from exc

    return image_path.replace("\\", "/")


def predict_from_request(
    *,
    image_path: str,
    sample_id: str | None = None,
    request_id: str | None = None,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Backend service function for one-image OpenWaste-HR inference.
    """
    resolved_project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else get_project_root()
    )

    safe_image_path = validate_project_relative_image_path(
        project_root=resolved_project_root,
        image_path=image_path,
    )

    resolved_config_path = (
        Path(config_path)
        if config_path is not None
        else get_default_api_config_path(resolved_project_root)
    )

    response = run_api_wrapper_prediction(
        config_path=resolved_config_path,
        project_root=resolved_project_root,
        request_payload={
            "image_path": safe_image_path,
            "sample_id": sample_id,
        },
        request_id=request_id,
    )

    return response