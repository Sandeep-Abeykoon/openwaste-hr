from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
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
    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def load_pytorch_model(
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


def summarize_latencies_ms(values: list[float]) -> dict[str, float]:
    values_sorted = sorted(values)

    return {
        "mean_ms": float(mean(values_sorted)),
        "median_ms": float(median(values_sorted)),
        "min_ms": float(values_sorted[0]),
        "max_ms": float(values_sorted[-1]),
        "p95_ms": float(np.percentile(values_sorted, 95)),
    }


def benchmark_pytorch(
    *,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, float]:
    model.eval()

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

        latencies_ms: list[float] = []

        for _ in range(timed_runs):
            start = time.perf_counter()

            _ = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            latencies_ms.append((end - start) * 1000.0)

    return summarize_latencies_ms(latencies_ms)


def benchmark_onnxruntime(
    *,
    onnx_path: Path,
    input_array: np.ndarray,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: input_array})

    latencies_ms: list[float] = []

    for _ in range(timed_runs):
        start = time.perf_counter()

        _ = session.run(None, {input_name: input_array})

        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000.0)

    summary = summarize_latencies_ms(latencies_ms)
    summary["provider"] = "CPUExecutionProvider"

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch and ONNX inference latency for OpenWaste-HR."
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
        "--onnx",
        type=Path,
        default=Path("ml/outputs/stage_04_add_trashbox_clean_v1/model_v1.onnx"),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("ml/outputs/stage_04_add_trashbox_clean_v1/latency_benchmark_summary_v1.json"),
    )

    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--timed-runs", type=int, default=100)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cpu_device = torch.device("cpu")

    cpu_model, config, class_names = load_pytorch_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=cpu_device,
    )

    image_size = int(config["preprocessing"]["image_size"])

    torch.manual_seed(42)
    cpu_input = torch.randn(1, 3, image_size, image_size, device=cpu_device)

    pytorch_cpu_summary = benchmark_pytorch(
        model=cpu_model,
        input_tensor=cpu_input,
        device=cpu_device,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )

    onnx_summary = benchmark_onnxruntime(
        onnx_path=args.onnx,
        input_array=cpu_input.detach().cpu().numpy().astype(np.float32),
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )

    cuda_summary: dict[str, float] | None = None

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")

        cuda_model, _, _ = load_pytorch_model(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=cuda_device,
        )

        cuda_input = cpu_input.to(cuda_device)

        cuda_summary = benchmark_pytorch(
            model=cuda_model,
            input_tensor=cuda_input,
            device=cuda_device,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )

    summary = {
        "experiment": config["experiment"]["name"],
        "class_names": class_names,
        "input_shape": [1, 3, image_size, image_size],
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "checkpoint_path": str(args.checkpoint),
        "onnx_path": str(args.onnx),
        "checkpoint_size_mb": file_size_mb(args.checkpoint),
        "onnx_size_mb": file_size_mb(args.onnx),
        "pytorch_cpu_latency": pytorch_cpu_summary,
        "pytorch_cuda_latency": cuda_summary,
        "onnxruntime_cpu_latency": onnx_summary,
        "note": (
            "Latency is measured on a synthetic 1x3x224x224 tensor and excludes image loading, "
            "file I/O, and preprocessing time."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
