from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from train_image_classifier import build_model, build_transforms


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(
    *,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], list[str]]:
    config = read_yaml(config_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = list(checkpoint.get("class_names", config["labels"]["known_classes"]))

    model = build_model(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config, class_names


def load_image_tensor(
    *,
    image_path: Path,
    transform: Any,
    device: torch.device,
) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    return tensor


def compute_prediction(
    *,
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    temperature: float,
) -> dict[str, Any]:
    with torch.no_grad():
        logits = model(image_tensor)

        probabilities = F.softmax(logits, dim=1)
        confidence_tensor, predicted_index_tensor = torch.max(probabilities, dim=1)

        scaled_probabilities = F.softmax(logits / temperature, dim=1)
        scaled_confidence_tensor, _ = torch.max(scaled_probabilities, dim=1)

        energy_tensor = -torch.logsumexp(logits, dim=1)

    predicted_index = int(predicted_index_tensor.item())
    predicted_label = class_names[predicted_index]

    class_probabilities = {
        class_names[index]: float(probabilities[0, index].item())
        for index in range(len(class_names))
    }

    return {
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "raw_confidence": float(confidence_tensor.item()),
        "temperature_scaled_confidence": float(scaled_confidence_tensor.item()),
        "energy": float(energy_tensor.item()),
        "class_probabilities": class_probabilities,
    }


def apply_decision_policy(
    *,
    prediction: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    reject_config = policy["reject_method"]
    threshold = float(reject_config["threshold"])
    direction = str(reject_config["direction"])

    energy = float(prediction["energy"])
    fine_label = str(prediction["predicted_label"])

    if direction != "lower_is_known":
        raise ValueError(
            f"This script currently expects lower_is_known energy policy, got: {direction}"
        )

    accepted_as_known = energy <= threshold

    coarse_mapping = policy["coarse_mapping"]
    coarse_label = coarse_mapping.get(fine_label, "unknown")

    if accepted_as_known:
        output_config = policy["decision_outputs"]["accept_known"]

        user_message = output_config["user_message_template"].format(
            fine_label=fine_label,
            coarse_label=coarse_label,
        )

        decision = {
            "decision": output_config["decision"],
            "accepted_as_known": True,
            "show_fine_label": True,
            "show_coarse_label": True,
            "fine_label": fine_label,
            "coarse_label": coarse_label,
            "user_message": user_message,
        }

    else:
        output_config = policy["decision_outputs"]["reject_unknown"]

        decision = {
            "decision": output_config["decision"],
            "accepted_as_known": False,
            "show_fine_label": False,
            "show_coarse_label": False,
            "fine_label": None,
            "coarse_label": None,
            "internal_top1_prediction": fine_label,
            "user_message": output_config["user_message_template"],
        }

    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenWaste-HR single-image inference with final decision policy."
    )

    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/configs/train_stage_04_add_trashbox_clean.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ml/outputs/stage_04_add_trashbox_clean_v1/best_model.pt"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("ml/configs/final_decision_policy.yaml"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config, class_names = load_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    policy = read_yaml(args.policy)

    transform = build_transforms(
        Path(config["preprocessing"]["config_path"]),
        train=False,
    )

    temperature = float(policy["calibration"]["temperature"])

    image_tensor = load_image_tensor(
        image_path=args.image,
        transform=transform,
        device=device,
    )

    prediction = compute_prediction(
        model=model,
        image_tensor=image_tensor,
        class_names=class_names,
        temperature=temperature,
    )

    decision = apply_decision_policy(
        prediction=prediction,
        policy=policy,
    )

    result = {
        "image_path": str(args.image),
        "model_config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "policy": str(args.policy),
        "device": str(device),
        "prediction": prediction,
        "decision": decision,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
