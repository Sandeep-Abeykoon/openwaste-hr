from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageOps
from torchvision import transforms

from openwaste_hr.evaluation.hierarchical_decision import (
    apply_hierarchical_policy,
)


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML dictionary.")

    return config


def resolve_path(project_root: str | Path, path_text: str | Path) -> Path:
    """
    Resolve a project-relative path.
    """
    return Path(project_root) / Path(path_text)


def get_device() -> torch.device:
    """
    Select CUDA when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Remove common prefixes added by wrappers such as DataParallel.
    """
    normalized: dict[str, Any] = {}

    for key, value in state_dict.items():
        new_key = str(key)

        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]

        normalized[new_key] = value

    return normalized


def extract_state_dict_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    """
    Extract model weights from different checkpoint formats.
    """
    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "weights",
        ]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return normalize_state_dict_keys(checkpoint[key])

        tensor_values = [
            value
            for value in checkpoint.values()
            if torch.is_tensor(value)
        ]

        if tensor_values:
            return normalize_state_dict_keys(checkpoint)

    raise ValueError("Could not extract a model state_dict from checkpoint.")


def resolve_class_names(
    checkpoint: Any,
    config_class_names: list[str],
) -> list[str]:
    """
    Resolve class names from checkpoint if available, otherwise use config.
    """
    if isinstance(checkpoint, dict):
        if "class_names" in checkpoint and isinstance(checkpoint["class_names"], list):
            return [str(label) for label in checkpoint["class_names"]]

        if "idx_to_class" in checkpoint and isinstance(checkpoint["idx_to_class"], dict):
            idx_to_class = checkpoint["idx_to_class"]

            return [
                str(idx_to_class[index])
                for index in sorted(idx_to_class, key=lambda item: int(item))
            ]

        if "class_to_idx" in checkpoint and isinstance(checkpoint["class_to_idx"], dict):
            class_to_idx = checkpoint["class_to_idx"]

            return [
                str(label)
                for label, _ in sorted(
                    class_to_idx.items(),
                    key=lambda item: int(item[1]),
                )
            ]

    if not config_class_names:
        raise ValueError("No class names found in checkpoint or config.")

    return [str(label) for label in config_class_names]


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
) -> torch.nn.Module:
    """
    Build the image classification model.

    Uses timm because the baseline model was trained using a timm-style model name.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for inference. Install it using requirements.txt."
        ) from exc

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


def load_model_for_inference(
    checkpoint_path: str | Path,
    model_name: str,
    num_classes: int,
    pretrained: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    """
    Load model and checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = extract_state_dict_from_checkpoint(checkpoint)

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, checkpoint


def build_image_transform(image_size: int) -> transforms.Compose:
    """
    Build inference transform.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_image_tensor(
    image_path: str | Path,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load a single image and convert it into a model-ready tensor.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")

    transform = build_image_transform(image_size)
    image_tensor = transform(image).unsqueeze(0).to(device)

    return image_tensor


def build_prediction_dataframe(
    sample_id: str,
    image_path: str,
    class_names: list[str],
    probabilities: list[float],
) -> pd.DataFrame:
    """
    Convert inference probabilities into the same format used by evaluation.
    """
    if len(class_names) != len(probabilities):
        raise ValueError("class_names and probabilities must have the same length.")

    max_probability = max(probabilities)
    predicted_index = probabilities.index(max_probability)
    predicted_label = class_names[predicted_index]

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "image_path": image_path,
        "pred_label": predicted_label,
        "max_softmax_confidence": round(float(max_probability), 6),
    }

    for class_name, probability in zip(class_names, probabilities):
        row[f"prob_{class_name}"] = round(float(probability), 6)

    return pd.DataFrame([row])


def run_model_prediction(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
) -> list[float]:
    """
    Run model and return softmax probabilities.
    """
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Model output size does not match the number of class names."
        )

    return [float(probability) for probability in probabilities]


def build_inference_result_payload(
    decisions_df: pd.DataFrame,
    class_names: list[str],
    probabilities: list[float],
    policy_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    Build JSON-safe inference result.
    """
    row = decisions_df.iloc[0].to_dict()

    probability_payload = {
        class_name: round(float(probability), 6)
        for class_name, probability in zip(class_names, probabilities)
    }

    payload = {
        "sample_id": str(row["sample_id"]),
        "image_path": str(row["image_path"]),
        "device": str(device),
        "pred_label": str(row["pred_label"]),
        "max_softmax_confidence": round(float(row["max_softmax_confidence"]), 6),
        "top_coarse_label": str(row["top_coarse_label"]),
        "top_coarse_confidence": round(float(row["top_coarse_confidence"]), 6),
        "second_coarse_confidence": round(
            float(row["second_coarse_confidence"]),
            6,
        ),
        "coarse_margin": round(float(row["coarse_margin"]), 6),
        "hierarchical_decision_type": str(row["hierarchical_decision_type"]),
        "hierarchical_final_label": str(row["hierarchical_final_label"]),
        "hierarchical_final_confidence": round(
            float(row["hierarchical_final_confidence"]),
            6,
        ),
        "hierarchical_decision_reason": str(row["hierarchical_decision_reason"]),
        "class_probabilities": probability_payload,
        "policy": {
            key: float(value)
            for key, value in policy_config.items()
        },
    }

    return payload


def write_markdown_result(
    output_path: Path,
    result_payload: dict[str, Any],
) -> None:
    """
    Write a small Markdown inference report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probability_rows = []

    for label, probability in result_payload["class_probabilities"].items():
        probability_rows.append(f"| {label} | {probability} |")

    probability_table = "\n".join(
        [
            "| Fine Label | Probability |",
            "|---|---:|",
            *probability_rows,
        ]
    )

    report = f"""# Single Image Inference Result v1

## Input

| Field | Value |
|---|---|
| sample_id | {result_payload["sample_id"]} |
| image_path | {result_payload["image_path"]} |
| device | {result_payload["device"]} |

## Model Prediction

| Field | Value |
|---|---|
| pred_label | {result_payload["pred_label"]} |
| max_softmax_confidence | {result_payload["max_softmax_confidence"]} |
| top_coarse_label | {result_payload["top_coarse_label"]} |
| top_coarse_confidence | {result_payload["top_coarse_confidence"]} |
| coarse_margin | {result_payload["coarse_margin"]} |

## Final OpenWaste-HR Decision

| Field | Value |
|---|---|
| hierarchical_decision_type | {result_payload["hierarchical_decision_type"]} |
| hierarchical_final_label | {result_payload["hierarchical_final_label"]} |
| hierarchical_final_confidence | {result_payload["hierarchical_final_confidence"]} |
| hierarchical_decision_reason | {result_payload["hierarchical_decision_reason"]} |

## Fine-Label Probabilities

{probability_table}

## Interpretation

The final decision is produced using the safe hierarchical policy. The system returns a fine label only when fine confidence is strong, falls back to a coarse label when broader evidence is stable, and otherwise routes the image to manual review.
"""

    output_path.write_text(report, encoding="utf-8")


def run_single_image_inference(
    config_path: str | Path,
    project_root: str | Path,
    image_path: str | Path,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """
    Run single-image OpenWaste-HR inference.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    model_config = config["model"]
    labels_config = config["labels"]
    policy_config = config["policy"]
    outputs_config = config["outputs"]

    resolved_image_path = resolve_path(project_root, image_path)

    if sample_id is None:
        sample_id = resolved_image_path.stem

    checkpoint_path = resolve_path(project_root, model_config["checkpoint_path"])
    output_metrics_dir = resolve_path(project_root, outputs_config["output_metrics_dir"])
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    model, checkpoint = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        model_name=str(model_config["model_name"]),
        num_classes=int(model_config["num_classes"]),
        pretrained=bool(model_config["pretrained"]),
        device=device,
    )

    class_names = resolve_class_names(
        checkpoint=checkpoint,
        config_class_names=list(model_config["class_names"]),
    )

    image_tensor = load_image_tensor(
        image_path=resolved_image_path,
        image_size=int(model_config["image_size"]),
        device=device,
    )

    probabilities = run_model_prediction(
        model=model,
        image_tensor=image_tensor,
        class_names=class_names,
    )

    prediction_df = build_prediction_dataframe(
        sample_id=str(sample_id),
        image_path=str(image_path),
        class_names=class_names,
        probabilities=probabilities,
    )

    decisions_df = apply_hierarchical_policy(
        predictions_df=prediction_df,
        fine_to_coarse={
            str(fine_label): str(coarse_label)
            for fine_label, coarse_label in labels_config["fine_to_coarse"].items()
        },
        fine_confidence_threshold=float(policy_config["fine_confidence_threshold"]),
        coarse_confidence_threshold=float(policy_config["coarse_confidence_threshold"]),
        coarse_margin_threshold=float(policy_config["coarse_margin_threshold"]),
        minimum_confidence_for_coarse=float(
            policy_config["minimum_confidence_for_coarse"]
        ),
    )

    result_payload = build_inference_result_payload(
        decisions_df=decisions_df,
        class_names=class_names,
        probabilities=probabilities,
        policy_config=policy_config,
        device=device,
    )

    json_path = output_metrics_dir / outputs_config["output_json"]
    markdown_path = output_metrics_dir / outputs_config["output_markdown"]

    json_path.write_text(
        json.dumps(result_payload, indent=2),
        encoding="utf-8",
    )

    write_markdown_result(
        output_path=markdown_path,
        result_payload=result_payload,
    )

    print("Single image inference completed successfully.")
    print("\nResult:")
    print(json.dumps(result_payload, indent=2))
    print("\nCreated files:")
    print(f"- JSON result: {json_path}")
    print(f"- Markdown result: {markdown_path}")

    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenWaste-HR inference on a single image."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/inference_pipeline.yaml",
        help="Path to inference pipeline YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Project-relative image path.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional sample ID for the image.",
    )

    args = parser.parse_args()

    run_single_image_inference(
        config_path=args.config,
        project_root=args.project_root,
        image_path=args.image,
        sample_id=args.sample_id,
    )


if __name__ == "__main__":
    main()