from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from openwaste_hr.evaluation.classification_metrics import build_prediction_dataframe
from openwaste_hr.evaluation.selective_metrics import (
    apply_confidence_threshold,
    compute_selective_metrics,
    select_threshold_from_sweep,
    sweep_confidence_thresholds,
)
from openwaste_hr.models.baseline_cnn import create_baseline_cnn
from openwaste_hr.training.image_transforms import build_eval_transform
from openwaste_hr.training.label_encoding import build_label_to_id
from openwaste_hr.training.torch_dataset import TorchManifestImageDataset


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


def get_device(device_name: str) -> torch.device:
    """
    Resolve device.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def build_threshold_list(
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
) -> list[float]:
    """
    Build threshold values from start to end inclusive.
    """
    if threshold_start < 0.0 or threshold_end > 1.0:
        raise ValueError("Threshold range must be between 0.0 and 1.0.")

    if threshold_step <= 0.0:
        raise ValueError("threshold_step must be greater than 0.0.")

    if threshold_start > threshold_end:
        raise ValueError("threshold_start must be less than or equal to threshold_end.")

    count = int(math.floor((threshold_end - threshold_start) / threshold_step)) + 1

    thresholds = [
        round(threshold_start + index * threshold_step, 6)
        for index in range(count)
    ]

    if thresholds[-1] < threshold_end:
        thresholds.append(round(threshold_end, 6))

    return thresholds


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
):
    """
    Load trained baseline model and class labels from checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run baseline training before confidence-reject evaluation."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    label_names = checkpoint["label_names"]
    train_config = checkpoint["config"]
    model_config = train_config["model"]

    model = create_baseline_cnn(
        model_name=str(model_config["name"]),
        num_classes=len(label_names),
        pretrained=False,
        drop_rate=float(model_config["drop_rate"]),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, label_names, checkpoint


def build_eval_loader(
    manifest_path: Path,
    project_root: Path,
    label_to_id: dict[str, int],
    label_column: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    Build evaluation dataloader from a manifest.
    """
    dataset = TorchManifestImageDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        label_to_id=label_to_id,
        usage_filter=None,
        label_column=label_column,
        transform=build_eval_transform(image_size=image_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


@torch.no_grad()
def run_prediction_loop(
    model,
    dataloader: DataLoader,
    device: torch.device,
    label_names: list[str],
) -> pd.DataFrame:
    """
    Run model predictions and return a prediction DataFrame.
    """
    model.eval()

    metadata_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    probability_rows: list[list[float]] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        max_probs, predictions = torch.max(probabilities, dim=1)

        batch_size = images.size(0)

        for index in range(batch_size):
            metadata_rows.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "image_path": str(batch["image_path"][index]),
                    "source_dataset": str(batch["source_dataset"][index]),
                    "source_split": str(batch["source_split"][index]),
                    "original_label": str(batch["original_label"][index]),
                    "fine_label": str(batch["fine_label"][index]),
                    "coarse_label": str(batch["coarse_label"][index]),
                    "usage": str(batch["usage"][index]),
                }
            )

        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        confidences.extend(max_probs.detach().cpu().tolist())
        probability_rows.extend(probabilities.detach().cpu().tolist())

    return build_prediction_dataframe(
        metadata_rows=metadata_rows,
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        probability_rows=probability_rows,
        label_names=label_names,
    )


def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    """
    Convert a DataFrame to Markdown without requiring tabulate.
    """
    if dataframe.empty:
        return "_No rows._"

    columns = list(dataframe.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    body_lines = []
    for _, row in dataframe.iterrows():
        values = [str(row[column]) for column in columns]
        body_lines.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *body_lines])


def plot_coverage_risk(
    sweep_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot validation coverage-risk curve.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        sweep_df["coverage"],
        sweep_df["selective_error_rate"],
        marker="o",
        markersize=3,
    )
    axis.set_title("Confidence-Threshold Coverage-Risk Curve")
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective Error Rate")
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_confidence_histogram(
    predictions_df: pd.DataFrame,
    selected_threshold: float,
    output_path: Path,
) -> None:
    """
    Plot confidence distribution for the test set.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.hist(predictions_df["max_softmax_confidence"].astype(float), bins=20)
    axis.axvline(selected_threshold, linestyle="--")
    axis.set_title("Test Confidence Distribution")
    axis.set_xlabel("Maximum Softmax Confidence")
    axis.set_ylabel("Image Count")
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_markdown_report(
    output_path: Path,
    label_names: list[str],
    selected_threshold: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    coverage_risk_plot_relative: str,
    confidence_histogram_relative: str,
) -> None:
    """
    Write thesis-ready confidence-reject report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_df = pd.DataFrame(
        [
            {
                "item": key,
                "value": value,
            }
            for key, value in selected_threshold.items()
        ]
    )

    val_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in val_metrics.items()
        ]
    )

    test_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in test_metrics.items()
        ]
    )

    report = f"""# Confidence-Threshold Reject Baseline v1

## Purpose

This report evaluates a simple reject-option baseline for OpenWaste-HR.

The trained closed-set classifier is allowed to reject low-confidence predictions instead of forcing every image into a known fine label.

## Classes

{", ".join(label_names)}

## Selected Threshold

The threshold was selected using the validation split only.

{dataframe_to_markdown_table(selected_df)}

## Validation Metrics After Rejection

{dataframe_to_markdown_table(val_df)}

## Test Metrics After Rejection

{dataframe_to_markdown_table(test_df)}

## Coverage-Risk Curve

![Coverage-risk curve]({coverage_risk_plot_relative})

## Test Confidence Histogram

![Confidence histogram]({confidence_histogram_relative})

## Research Interpretation

This confidence-threshold baseline is the first safety-oriented baseline after the closed-set classifier.

It measures whether low-confidence predictions can be routed to manual review. This is important because the final OpenWaste-HR system should not only classify known items, but also reduce unsafe confident errors by rejecting uncertain, ambiguous, or unknown inputs.

This is still not the final proposed model. Later stages will add open-set scoring and hierarchical coarse/fine fallback.
"""

    output_path.write_text(report, encoding="utf-8")


def run_confidence_reject_baseline(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """
    Run confidence-threshold reject baseline.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    data_config = config["data"]
    threshold_config = config["threshold_selection"]
    output_config = config["outputs"]

    checkpoint_path = resolve_path(project_root, paths["checkpoint"])
    val_manifest_path = resolve_path(project_root, paths["val_manifest"])
    test_manifest_path = resolve_path(project_root, paths["test_manifest"])

    output_metrics_dir = resolve_path(project_root, paths["output_metrics_dir"])
    output_figures_dir = resolve_path(project_root, paths["output_figures_dir"])
    docs_results_dir = resolve_path(project_root, paths["docs_results_dir"])
    docs_figures_dir = resolve_path(project_root, paths["docs_figures_dir"])

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    docs_results_dir.mkdir(parents=True, exist_ok=True)
    docs_figures_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(str(data_config["device"]))

    model, label_names, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    label_to_id = build_label_to_id(label_names)

    val_loader = build_eval_loader(
        manifest_path=val_manifest_path,
        project_root=project_root,
        label_to_id=label_to_id,
        label_column=str(data_config["label_column"]),
        image_size=int(data_config["image_size"]),
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
    )

    test_loader = build_eval_loader(
        manifest_path=test_manifest_path,
        project_root=project_root,
        label_to_id=label_to_id,
        label_column=str(data_config["label_column"]),
        image_size=int(data_config["image_size"]),
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
    )

    val_predictions_df = run_prediction_loop(
        model=model,
        dataloader=val_loader,
        device=device,
        label_names=label_names,
    )

    test_predictions_df = run_prediction_loop(
        model=model,
        dataloader=test_loader,
        device=device,
        label_names=label_names,
    )

    thresholds = build_threshold_list(
        threshold_start=float(threshold_config["threshold_start"]),
        threshold_end=float(threshold_config["threshold_end"]),
        threshold_step=float(threshold_config["threshold_step"]),
    )

    sweep_df = sweep_confidence_thresholds(
        predictions_df=val_predictions_df,
        label_names=label_names,
        thresholds=thresholds,
    )

    selected_threshold = select_threshold_from_sweep(
        sweep_df=sweep_df,
        min_coverage=float(threshold_config["min_coverage"]),
        selection_metric=str(threshold_config["selection_metric"]),
    )

    threshold_value = float(selected_threshold["threshold"])

    val_thresholded_df = apply_confidence_threshold(
        predictions_df=val_predictions_df,
        threshold=threshold_value,
    )

    test_thresholded_df = apply_confidence_threshold(
        predictions_df=test_predictions_df,
        threshold=threshold_value,
    )

    val_metrics = compute_selective_metrics(
        thresholded_df=val_thresholded_df,
        label_names=label_names,
    )

    test_metrics = compute_selective_metrics(
        thresholded_df=test_thresholded_df,
        label_names=label_names,
    )

    val_predictions_path = output_metrics_dir / output_config["val_predictions_csv"]
    test_predictions_path = output_metrics_dir / output_config["test_predictions_csv"]
    threshold_sweep_path = output_metrics_dir / output_config["threshold_sweep_csv"]
    selected_threshold_path = output_metrics_dir / output_config["selected_threshold_json"]
    test_metrics_path = output_metrics_dir / output_config["test_metrics_json"]

    val_thresholded_df.to_csv(val_predictions_path, index=False)
    test_thresholded_df.to_csv(test_predictions_path, index=False)
    sweep_df.to_csv(threshold_sweep_path, index=False)

    selected_threshold_payload = {
        "selected_threshold": selected_threshold,
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "label_names": label_names,
    }

    selected_threshold_path.write_text(
        json.dumps(selected_threshold_payload, indent=2),
        encoding="utf-8",
    )

    test_metrics_payload = {
        "selected_threshold": threshold_value,
        "validation_metrics_after_rejection": val_metrics,
        "test_metrics_after_rejection": test_metrics,
        "label_names": label_names,
    }

    test_metrics_path.write_text(
        json.dumps(test_metrics_payload, indent=2),
        encoding="utf-8",
    )

    coverage_risk_plot_path = output_figures_dir / output_config["coverage_risk_plot"]
    confidence_histogram_path = output_figures_dir / output_config["confidence_histogram_plot"]

    plot_coverage_risk(
        sweep_df=sweep_df,
        output_path=coverage_risk_plot_path,
    )

    plot_confidence_histogram(
        predictions_df=test_predictions_df,
        selected_threshold=threshold_value,
        output_path=confidence_histogram_path,
    )

    docs_coverage_risk_path = docs_figures_dir / output_config["coverage_risk_plot"]
    docs_confidence_histogram_path = docs_figures_dir / output_config["confidence_histogram_plot"]

    shutil.copyfile(coverage_risk_plot_path, docs_coverage_risk_path)
    shutil.copyfile(confidence_histogram_path, docs_confidence_histogram_path)

    markdown_report_path = docs_results_dir / output_config["markdown_report"]

    write_markdown_report(
        output_path=markdown_report_path,
        label_names=label_names,
        selected_threshold=selected_threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        coverage_risk_plot_relative=f"figures/{output_config['coverage_risk_plot']}",
        confidence_histogram_relative=f"figures/{output_config['confidence_histogram_plot']}",
    )

    print("Confidence-threshold reject baseline completed successfully.")
    print(f"Device: {device}")
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Selected threshold: {threshold_value:.4f}")
    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nCreated files:")
    print(f"- validation predictions: {val_predictions_path}")
    print(f"- test predictions: {test_predictions_path}")
    print(f"- threshold sweep: {threshold_sweep_path}")
    print(f"- selected threshold: {selected_threshold_path}")
    print(f"- test metrics: {test_metrics_path}")
    print(f"- coverage-risk plot: {coverage_risk_plot_path}")
    print(f"- confidence histogram: {confidence_histogram_path}")
    print(f"- thesis report: {markdown_report_path}")

    return {
        "selected_threshold": threshold_value,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "markdown_report": str(markdown_report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run confidence-threshold reject baseline."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/confidence_reject_baseline_trashnet.yaml",
        help="Path to confidence reject YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    run_confidence_reject_baseline(
        config_path=args.config,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()