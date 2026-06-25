from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from train_image_classifier import build_model


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


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


def verify_onnx_output(
    *,
    model: torch.nn.Module,
    onnx_path: Path,
    dummy_input: torch.Tensor,
) -> dict[str, float]:
    import onnxruntime as ort

    with torch.no_grad():
        torch_output = model(dummy_input).detach().cpu().numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    onnx_output = session.run(
        None,
        {input_name: dummy_input.detach().cpu().numpy()},
    )[0]

    abs_diff = np.abs(torch_output - onnx_output)

    return {
        "max_abs_difference": float(np.max(abs_diff)),
        "mean_abs_difference": float(np.mean(abs_diff)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the final OpenWaste-HR model to ONNX."
    )

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
        "--output-onnx",
        type=Path,
        default=Path("ml/outputs/stage_04_add_trashbox_clean_v1/model_v1.onnx"),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("ml/outputs/stage_04_add_trashbox_clean_v1/onnx_export_summary_v1.json"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")

    model, config, class_names = load_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    image_size = int(config["preprocessing"]["image_size"])
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        args.output_onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    verification = verify_onnx_output(
        model=model,
        onnx_path=args.output_onnx,
        dummy_input=dummy_input,
    )

    summary = {
        "model": config["experiment"]["name"],
        "checkpoint": str(args.checkpoint),
        "onnx_path": str(args.output_onnx),
        "class_names": class_names,
        "input_shape": [1, 3, image_size, image_size],
        "opset_version": 17,
        "checkpoint_size_mb": file_size_mb(args.checkpoint),
        "onnx_size_mb": file_size_mb(args.output_onnx),
        "verification": verification,
        "status": "exported_and_verified",
    }

    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
