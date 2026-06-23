from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from train_image_classifier import build_model, build_transforms


FUSION_FEATURES = [
    "confidence",
    "temperature_scaled_confidence",
    "max_logit",
    "energy",
    "softmax_margin",
    "softmax_entropy",
    "mahalanobis_knownness",
]


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_classifier(
    *,
    training_config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = list(
        checkpoint.get(
            "class_names",
            training_config["labels"]["known_classes"],
        )
    )

    model = build_model(training_config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def find_last_linear_layer(model: nn.Module) -> tuple[str, nn.Linear]:
    last_name = ""
    last_layer: nn.Linear | None = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_layer = module

    if last_layer is None:
        raise RuntimeError("Could not find the final Linear layer for embedding extraction.")

    return last_name, last_layer


def preprocess_image(
    *,
    image_path: Path,
    transform: Any,
    device: torch.device,
) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    return tensor


def run_classifier_with_embedding(
    *,
    model: nn.Module,
    image_tensor: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, str]:
    layer_name, last_linear = find_last_linear_layer(model)

    captured_embeddings: list[torch.Tensor] = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured_embeddings.append(inputs[0].detach().cpu())

    handle = last_linear.register_forward_pre_hook(hook)

    try:
        with torch.no_grad():
            logits = model(image_tensor)

    finally:
        handle.remove()

    if not captured_embeddings:
        raise RuntimeError("Embedding hook did not capture any features.")

    embedding = captured_embeddings[0].numpy()[0]

    return logits.detach().cpu(), embedding, layer_name


def compute_model_scores(
    *,
    logits: torch.Tensor,
    class_names: list[str],
    temperature: float,
) -> dict[str, Any]:
    probabilities = F.softmax(logits, dim=1)[0]
    temperature_probabilities = F.softmax(logits / temperature, dim=1)[0]

    confidence, pred_index_tensor = torch.max(probabilities, dim=0)
    temperature_confidence, _ = torch.max(temperature_probabilities, dim=0)
    max_logit, _ = torch.max(logits[0], dim=0)
    energy = -torch.logsumexp(logits[0], dim=0)

    pred_index = int(pred_index_tensor.item())
    pred_label = class_names[pred_index]

    probs_np = probabilities.numpy().astype(float)
    sorted_probs = np.sort(probs_np)[::-1]

    softmax_margin = float(sorted_probs[0] - sorted_probs[1])
    softmax_entropy = float(
        -np.sum(probs_np * np.log(np.clip(probs_np, 1e-12, 1.0)))
    )

    scores: dict[str, Any] = {
        "pred_label": pred_label,
        "pred_index": pred_index,
        "confidence": float(confidence.item()),
        "temperature_scaled_confidence": float(temperature_confidence.item()),
        "max_logit": float(max_logit.item()),
        "energy": float(energy.item()),
        "softmax_margin": softmax_margin,
        "softmax_entropy": softmax_entropy,
    }

    for class_index, class_name in enumerate(class_names):
        scores[f"prob_{class_name}"] = float(probabilities[class_index].item())

    return scores


def compute_mahalanobis_knownness(
    *,
    embedding: np.ndarray,
    mahalanobis_model: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    scaler = mahalanobis_model["scaler"]
    class_means = mahalanobis_model["class_means"]
    precision = mahalanobis_model["precision"]

    embedding_z = scaler.transform(embedding.reshape(1, -1))[0]

    distances: list[float] = []

    for class_name in class_names:
        mean = class_means[class_name]
        diff = embedding_z - mean
        distance = float(diff.T @ precision @ diff)
        distance = max(distance, 0.0)
        distances.append(distance)

    nearest_index = int(np.argmin(distances))
    nearest_class = class_names[nearest_index]
    min_distance = float(distances[nearest_index])

    return {
        "mahalanobis_min_distance": min_distance,
        "mahalanobis_knownness": float(-min_distance),
        "mahalanobis_nearest_class": nearest_class,
    }


def build_fusion_feature_vector(scores: dict[str, Any]) -> np.ndarray:
    values = [float(scores[feature_name]) for feature_name in FUSION_FEATURES]
    return np.array(values, dtype=float).reshape(1, -1)


def make_user_message(
    *,
    accepted: bool,
    pred_label: str,
    coarse_label: str,
) -> str:
    if accepted:
        return (
            f"This item is likely {pred_label}. "
            f"It belongs to the {coarse_label} category."
        )

    return (
        "The system is not confident that this item belongs to the supported "
        "recyclable classes. Please send it for manual review."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenWaste-HR final Fusion Gate v2 inference on one image."
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image to classify.",
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("ml/configs/train_stage_04_add_trashbox_clean.yaml"),
        help="Training config used for the Stage 4 classifier.",
    )

    parser.add_argument(
        "--policy-config",
        type=Path,
        default=Path("ml/configs/final_decision_policy_v2_fusion_gate.yaml"),
        help="Final Fusion Gate v2 policy config.",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the inference result JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    training_config = read_yaml(args.training_config)
    policy_config = read_yaml(args.policy_config)

    checkpoint_path = Path(policy_config["base_model"]["checkpoint_path"])
    mahalanobis_model_path = Path(policy_config["mahalanobis"]["model_path"])
    fusion_gate_model_path = Path(policy_config["fusion_gate"]["model_path"])

    threshold = float(policy_config["fusion_gate"]["threshold"])
    temperature = float(policy_config["temperature_scaling"]["temperature"])

    if not mahalanobis_model_path.exists():
        raise FileNotFoundError(f"Mahalanobis model not found: {mahalanobis_model_path}")

    if not fusion_gate_model_path.exists():
        raise FileNotFoundError(f"Fusion gate model not found: {fusion_gate_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names = load_classifier(
        training_config=training_config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    transform = build_transforms(
        Path(training_config["preprocessing"]["config_path"]),
        train=False,
    )

    image_tensor = preprocess_image(
        image_path=args.image,
        transform=transform,
        device=device,
    )

    logits, embedding, embedding_layer = run_classifier_with_embedding(
        model=model,
        image_tensor=image_tensor,
    )

    model_scores = compute_model_scores(
        logits=logits,
        class_names=class_names,
        temperature=temperature,
    )

    mahalanobis_model = joblib.load(mahalanobis_model_path)

    mahalanobis_scores = compute_mahalanobis_knownness(
        embedding=embedding,
        mahalanobis_model=mahalanobis_model,
        class_names=class_names,
    )

    combined_scores = {
        **model_scores,
        **mahalanobis_scores,
    }

    fusion_gate = joblib.load(fusion_gate_model_path)
    fusion_feature_vector = build_fusion_feature_vector(combined_scores)

    fusion_knownness_score = float(fusion_gate.predict_proba(fusion_feature_vector)[0, 1])
    accepted = bool(fusion_knownness_score >= threshold)

    decision_type = "known_fine_label" if accepted else "unknown_manual_review"
    coarse_label = "recyclable" if accepted else "manual_review_required"

    result = {
        "policy_version": policy_config["policy_version"],
        "image_path": str(args.image),
        "device": str(device),
        "embedding_layer": embedding_layer,
        "embedding_dimension": int(embedding.shape[0]),
        "known_classes": class_names,
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
                for class_name in class_names
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
            "threshold": threshold,
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

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
