from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from openwaste_hr.inference.single_image_inference import (
    load_yaml,
    resolve_path,
    run_single_image_inference,
)


def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No rows._"

    columns = list(dataframe.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in dataframe.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def normalize_extension(extension: str) -> str:
    extension = str(extension).strip().lower()

    if not extension.startswith("."):
        extension = f".{extension}"

    return extension


def validate_api_request(
    request_payload: dict[str, Any],
    allowed_image_extensions: list[str],
    max_image_path_length: int,
) -> dict[str, str | None]:
    """
    Validate and normalize an API-style inference request.
    """
    if not isinstance(request_payload, dict):
        raise ValueError("API request payload must be a dictionary.")

    image_path = str(request_payload.get("image_path", "")).strip()
    sample_id_raw = request_payload.get("sample_id", None)

    if not image_path:
        raise ValueError("image_path is required.")

    if len(image_path) > int(max_image_path_length):
        raise ValueError("image_path is too long.")

    allowed_extensions = {
        normalize_extension(extension)
        for extension in allowed_image_extensions
    }

    image_extension = Path(image_path).suffix.lower()

    if image_extension not in allowed_extensions:
        raise ValueError(
            f"Unsupported image extension: {image_extension}. "
            f"Allowed extensions: {sorted(allowed_extensions)}"
        )

    sample_id: str | None

    if sample_id_raw is None:
        sample_id = None
    else:
        sample_id = str(sample_id_raw).strip()

        if not sample_id:
            sample_id = None

    return {
        "image_path": image_path,
        "sample_id": sample_id,
    }


def build_success_response(
    inference_result: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """
    Convert raw inference output into a backend-friendly response.
    """
    return {
        "status": "success",
        "request_id": request_id,
        "request": {
            "sample_id": inference_result["sample_id"],
            "image_path": inference_result["image_path"],
        },
        "prediction": {
            "pred_label": inference_result["pred_label"],
            "max_softmax_confidence": inference_result["max_softmax_confidence"],
            "top_coarse_label": inference_result["top_coarse_label"],
            "top_coarse_confidence": inference_result["top_coarse_confidence"],
            "coarse_margin": inference_result["coarse_margin"],
        },
        "decision": {
            "decision_type": inference_result["hierarchical_decision_type"],
            "final_label": inference_result["hierarchical_final_label"],
            "final_confidence": inference_result["hierarchical_final_confidence"],
            "reason": inference_result["hierarchical_decision_reason"],
        },
        "class_probabilities": inference_result["class_probabilities"],
        "policy": inference_result["policy"],
        "metadata": {
            "device": inference_result.get("device", "unknown"),
            "pipeline_version": "prototype_api_wrapper_v1",
        },
    }


def build_error_response(
    error: Exception,
    request_id: str,
) -> dict[str, Any]:
    """
    Convert an exception into a stable API-style error response.
    """
    return {
        "status": "error",
        "request_id": request_id,
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
        "metadata": {
            "pipeline_version": "prototype_api_wrapper_v1",
        },
    }


def write_markdown_response(
    output_path: Path,
    response_payload: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if response_payload["status"] == "error":
        error = response_payload["error"]

        report = f"""# Prototype API Wrapper v1 Response

## Status

| Field | Value |
|---|---|
| status | error |
| request_id | {response_payload["request_id"]} |
| error_type | {error["type"]} |
| error_message | {error["message"]} |
"""

        output_path.write_text(report, encoding="utf-8")
        return

    request = response_payload["request"]
    prediction = response_payload["prediction"]
    decision = response_payload["decision"]

    probability_df = pd.DataFrame(
        [
            {
                "fine_label": label,
                "probability": probability,
            }
            for label, probability in response_payload["class_probabilities"].items()
        ]
    )

    report = f"""# Prototype API Wrapper v1 Response

## Status

| Field | Value |
|---|---|
| status | success |
| request_id | {response_payload["request_id"]} |

## Request

| Field | Value |
|---|---|
| sample_id | {request["sample_id"]} |
| image_path | {request["image_path"]} |

## Prediction

| Field | Value |
|---|---|
| pred_label | {prediction["pred_label"]} |
| max_softmax_confidence | {prediction["max_softmax_confidence"]} |
| top_coarse_label | {prediction["top_coarse_label"]} |
| top_coarse_confidence | {prediction["top_coarse_confidence"]} |
| coarse_margin | {prediction["coarse_margin"]} |

## Final Decision

| Field | Value |
|---|---|
| decision_type | {decision["decision_type"]} |
| final_label | {decision["final_label"]} |
| final_confidence | {decision["final_confidence"]} |
| reason | {decision["reason"]} |

## Class Probabilities

{dataframe_to_markdown_table(probability_df)}

## Research Interpretation

This response is a backend-friendly representation of the OpenWaste-HR inference result.
"""

    output_path.write_text(report, encoding="utf-8")


def run_api_wrapper_prediction(
    config_path: str | Path,
    project_root: str | Path,
    request_payload: dict[str, Any],
    request_id: str | None = None,
    raise_errors: bool = False,
) -> dict[str, Any]:
    """
    Run the prototype API wrapper.

    This function validates an API-style request, calls single-image inference,
    and writes a stable JSON/Markdown response.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    api_config = config["api"]
    inference_config = config["inference"]
    outputs_config = config["outputs"]

    if request_id is None:
        request_id = str(uuid.uuid4())

    output_metrics_dir = resolve_path(project_root, outputs_config["output_metrics_dir"])
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_metrics_dir / outputs_config["output_response_json"]
    markdown_path = output_metrics_dir / outputs_config["output_response_markdown"]

    try:
        normalized_request = validate_api_request(
            request_payload=request_payload,
            allowed_image_extensions=list(api_config["allowed_image_extensions"]),
            max_image_path_length=int(api_config["max_image_path_length"]),
        )

        inference_result = run_single_image_inference(
            config_path=resolve_path(
                project_root,
                inference_config["single_image_config_path"],
            ),
            project_root=project_root,
            image_path=str(normalized_request["image_path"]),
            sample_id=normalized_request["sample_id"],
        )

        response_payload = build_success_response(
            inference_result=inference_result,
            request_id=request_id,
        )

    except Exception as error:
        if raise_errors:
            raise

        response_payload = build_error_response(
            error=error,
            request_id=request_id,
        )

    json_path.write_text(
        json.dumps(response_payload, indent=2),
        encoding="utf-8",
    )

    write_markdown_response(
        output_path=markdown_path,
        response_payload=response_payload,
    )

    print("Prototype API wrapper completed.")
    print("\nResponse:")
    print(json.dumps(response_payload, indent=2))
    print("\nCreated files:")
    print(f"- JSON response: {json_path}")
    print(f"- Markdown response: {markdown_path}")

    return response_payload


def load_request_json(request_json_path: str | Path) -> dict[str, Any]:
    request_json_path = Path(request_json_path)

    if not request_json_path.exists():
        raise FileNotFoundError(f"Request JSON file not found: {request_json_path}")

    request_payload = json.loads(request_json_path.read_text(encoding="utf-8"))

    if not isinstance(request_payload, dict):
        raise ValueError("Request JSON must contain an object.")

    return request_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prototype API-style OpenWaste-HR inference."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/prototype_api_wrapper.yaml",
        help="Path to prototype API wrapper YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Project-relative image path.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional sample ID.",
    )
    parser.add_argument(
        "--request-json",
        default=None,
        help="Optional JSON request file. If supplied, it overrides --image.",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Optional request ID.",
    )

    args = parser.parse_args()

    if args.request_json:
        request_payload = load_request_json(args.request_json)
    else:
        if not args.image:
            raise ValueError("Either --image or --request-json must be provided.")

        request_payload = {
            "image_path": args.image,
            "sample_id": args.sample_id,
        }

    run_api_wrapper_prediction(
        config_path=args.config,
        project_root=args.project_root,
        request_payload=request_payload,
        request_id=args.request_id,
    )


if __name__ == "__main__":
    main()