# ONNX Export and Latency Benchmark Results v1

## Purpose

This report records the deployment-readiness evaluation for the final Stage 4 OpenWaste-HR model.

The goal was to export the trained PyTorch model to ONNX format and benchmark inference latency using:

- PyTorch CPU
- PyTorch CUDA
- ONNX Runtime CPU

The benchmark measures model inference time only. It excludes image loading, file I/O, and preprocessing.

## Model

| Item | Value |
|---|---|
| Experiment | stage_04_add_trashbox_clean_v1 |
| Known classes | cardboard, glass, metal, paper, plastic |
| Input shape | 1 x 3 x 224 x 224 |
| Checkpoint path | ml\outputs\stage_04_add_trashbox_clean_v1\best_model.pt |
| ONNX path | ml\outputs\stage_04_add_trashbox_clean_v1\model_v1.onnx |
| ONNX opset version | 17 |

## Export Verification

The ONNX model was verified by comparing its output logits against the PyTorch model output on the same dummy input.

| Verification metric | Value |
|---|---:|
| Maximum absolute difference | 0.000004768 |
| Mean absolute difference | 0.000002134 |
| Export status | exported_and_verified |

The very small output differences show that the ONNX export is numerically consistent with the PyTorch model for the tested input.

## Model Size

| Artifact | Size |
|---|---:|
| PyTorch checkpoint | 16.25 MB |
| ONNX model | 0.34 MB |

## Latency Benchmark Setup

| Item | Value |
|---|---:|
| Warm-up runs | 20 |
| Timed runs | 100 |
| Input tensor | synthetic 1 x 3 x 224 x 224 |
| ONNX provider | CPUExecutionProvider |

## Latency Results

| Runtime | Mean latency | Median latency | Min latency | Max latency | P95 latency |
|---|---:|---:|---:|---:|---:|
| PyTorch CPU | 20.07 ms | 19.57 ms | 18.65 ms | 24.01 ms | 23.72 ms |
| PyTorch CUDA | 8.38 ms | 8.27 ms | 7.72 ms | 9.59 ms | 9.30 ms |
| ONNX Runtime CPU | 4.95 ms | 4.41 ms | 4.20 ms | 34.41 ms | 5.34 ms |

## Interpretation

The ONNX Runtime CPU result was the fastest measured runtime, with a mean latency of 4.95 ms and a median latency of 4.41 ms. This was faster than both PyTorch CPU and PyTorch CUDA in the benchmark.

The PyTorch CUDA result was also fast, with a mean latency of 8.38 ms. However, ONNX Runtime CPU provided the best deployment latency for this single-image inference benchmark.

The PyTorch CPU result was slower, with a mean latency of 20.07 ms, but it is still suitable for interactive image-upload inference.

## Research Relevance

This benchmark supports the deployment-readiness part of the OpenWaste-HR project. The final model can be exported to ONNX and can perform fast single-image inference on a laptop environment.

This means the model is suitable for an upload-based or webcam-based prototype where the user submits an image and receives:

1. a known fine-label prediction if accepted,
2. a recyclable fallback label, or
3. an unknown/manual-review decision if rejected by the energy-based policy.

## Status

ONNX export and latency benchmarking are complete.

The next implementation step is to connect the final model and decision policy to a simple demo interface.
