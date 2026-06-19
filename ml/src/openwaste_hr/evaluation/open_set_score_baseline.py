from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from openwaste_hr.evaluation.classification_metrics import build_prediction_dataframe
from openwaste_hr.evaluation.open_set_scores import (
    apply_score_threshold,
    build_thresholds_from_scores,
    compute_energy_score,
    compute_max_logit_score,
    compute_score_selective_metrics,
    select_score_threshold_from_sweep,
    sweep_score_thresholds,
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
            "Run baseline training before open-set score evaluation."
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
    Build evaluation dataloader.
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
def run_score_prediction_loop(
    model,
    dataloader: DataLoader,
    device: torch.device,
    label_names: list[str],
    energy_temperature: float,
) -> pd.DataFrame:
    """
    Run model predictions and add open-set score columns.
    """
    model.eval()

    metadata_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    probability_rows: list[list[float]] = []

    max_logit_scores: list[float] = []
    energy_scores: list[float] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        max_probs, predictions = torch.max(probabilities, dim=1)

        batch_max_logit = compute_max_logit_score(logits)
        batch_energy = compute_energy_score(
            logits=logits,
            temperature=energy_temperature,
        )

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
        max_logit_scores.extend(batch_max_logit.detach().cpu().tolist())
        energy_scores.extend(batch_energy.detach().cpu().tolist())

    predictions_df = build_prediction_dataframe(
        metadata_rows=metadata_rows,
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        probability_rows=probability_rows,
        label_names=label_names,
    )

    predictions_df["max_logit_score"] = [
        round(float(value), 6)
        for value in max_logit_scores
    ]

    predictions_df["energy_score"] = [
        round(float(value), 6)
        for value in energy_scores
    ]

    return predictions_df


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
    Plot coverage-risk curves for max-logit and energy.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))

    for score_name, group in sweep_df.groupby("score_name", sort=True):
        group = group.sort_values("coverage")
        axis.plot(
            group["coverage"],
            group["selective_error_rate"],
            marker="o",
            markersize=3,
            label=score_name,
        )

    axis.set_title("Open-Set Score Coverage-Risk Curve")
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective Error Rate")
    axis.legend()
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_score_histograms(
    test_predictions_df: pd.DataFrame,
    selected_thresholds: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Plot score histograms for test set.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(test_predictions_df["max_logit_score"].astype(float), bins=20)
    if "max_logit" in selected_thresholds:
        axes[0].axvline(
            float(selected_thresholds["max_logit"]["threshold"]),
            linestyle="--",
        )
    axes[0].set_title("Maximum Logit Score")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Image Count")
    axes[0].grid(True)

    axes[1].hist(test_predictions_df["energy_score"].astype(float), bins=20)
    if "energy" in selected_thresholds:
        axes[1].axvline(
            float(selected_thresholds["energy"]["threshold"]),
            linestyle="--",
        )
    axes[1].set_title("Energy Score")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Image Count")
    axes[1].grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_markdown_report(
    output_path: Path,
    label_names: list[str],
    selected_thresholds: dict[str, Any],
    test_metrics: dict[str, Any],
    coverage_risk_plot_relative: str,
    score_histogram_relative: str,
) -> None:
    """
    Write thesis-ready open-set score baseline report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    for score_name, payload in selected_thresholds.items():
        selected_rows.append(
            {
                "score_name": score_name,
                "threshold": payload["threshold"],
                "accept_direction": payload["accept_direction"],
                "validation_coverage": payload["coverage"],
                "validation_selective_accuracy": payload["selective_accuracy"],
                "validation_selective_macro_f1": payload["selective_macro_f1"],
            }
        )

    metric_rows = []
    for score_name, payload in test_metrics.items():
        for metric_name, value in payload.items():
            metric_rows.append(
                {
                    "score_name": score_name,
                    "metric": metric_name,
                    "value": value,
                }
            )

    selected_df = pd.DataFrame(selected_rows)
    metrics_df = pd.DataFrame(metric_rows)

    report = f"""# Open-Set Score Reject Baseline v1

## Purpose

This report evaluates two logit-based reject baselines:

1. Maximum Logit Score
2. Energy Score

The goal is to compare these against the previous softmax-confidence reject baseline.

## Classes

{", ".join(label_names)}

## Selected Thresholds

Thresholds were selected using the validation split only.

{dataframe_to_markdown_table(selected_df)}

## Test Metrics

{dataframe_to_markdown_table(metrics_df)}

## Coverage-Risk Plot

![Open-set score coverage-risk plot]({coverage_risk_plot_relative})

## Score Histogram Plot

![Open-set score histogram plot]({score_histogram_relative})

## Research Interpretation

This stage prepares OpenWaste-HR for true open-set and unknown-item evaluation.

Maximum logit and energy scoring use raw model logits instead of only softmax confidence. This is important because later unknown and local difficult images may expose overconfident softmax behaviour.

Current limitation: this report still evaluates selective classification on known TrashNet test images. The next dataset stage must introduce held-out unknown and local unknown images to test true unknown detection.
"""

    output_path.write_text(report, encoding="utf-8")


def run_open_set_score_baseline(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """
    Run maximum-logit and energy-score reject baselines.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    data_config = config["data"]
    score_config = config["score_methods"]
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

    energy_temperature = float(score_config["energy"].get("temperature", 1.0))

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

    val_predictions_df = run_score_prediction_loop(
        model=model,
        dataloader=val_loader,
        device=device,
        label_names=label_names,
        energy_temperature=energy_temperature,
    )

    test_predictions_df = run_score_prediction_loop(
        model=model,
        dataloader=test_loader,
        device=device,
        label_names=label_names,
        energy_temperature=energy_temperature,
    )

    method_definitions = {
        "max_logit": {
            "score_column": "max_logit_score",
            "accept_direction": score_config["max_logit"]["accept_direction"],
            "threshold_count": int(score_config["max_logit"]["threshold_count"]),
            "enabled": bool(score_config["max_logit"]["enabled"]),
        },
        "energy": {
            "score_column": "energy_score",
            "accept_direction": score_config["energy"]["accept_direction"],
            "threshold_count": int(score_config["energy"]["threshold_count"]),
            "enabled": bool(score_config["energy"]["enabled"]),
        },
    }

    all_sweep_parts = []
    selected_thresholds: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    test_thresholded_parts = []

    for score_name, method in method_definitions.items():
        if not method["enabled"]:
            continue

        score_column = method["score_column"]

        thresholds = build_thresholds_from_scores(
            scores=val_predictions_df[score_column],
            threshold_count=int(method["threshold_count"]),
        )

        sweep_df = sweep_score_thresholds(
            predictions_df=val_predictions_df,
            label_names=label_names,
            score_name=score_name,
            score_column=score_column,
            thresholds=thresholds,
            accept_direction=str(method["accept_direction"]),
        )

        selected = select_score_threshold_from_sweep(
            sweep_df=sweep_df,
            min_coverage=float(threshold_config["min_coverage"]),
            selection_metric=str(threshold_config["selection_metric"]),
        )

        selected_thresholds[score_name] = selected

        thresholded_test_df = apply_score_threshold(
            predictions_df=test_predictions_df,
            score_column=score_column,
            threshold=float(selected["threshold"]),
            accept_direction=str(selected["accept_direction"]),
        )

        method_test_metrics = compute_score_selective_metrics(
            thresholded_df=thresholded_test_df,
            label_names=label_names,
        )

        test_metrics[score_name] = method_test_metrics

        thresholded_test_df.insert(0, "score_name", score_name)
        test_thresholded_parts.append(thresholded_test_df)

        all_sweep_parts.append(sweep_df)

    if not all_sweep_parts:
        raise ValueError("No score methods were enabled.")

    threshold_sweep_df = pd.concat(all_sweep_parts, ignore_index=True)
    test_thresholded_df = pd.concat(test_thresholded_parts, ignore_index=True)

    val_scores_path = output_metrics_dir / output_config["val_scores_csv"]
    test_scores_path = output_metrics_dir / output_config["test_scores_csv"]
    threshold_sweep_path = output_metrics_dir / output_config["threshold_sweep_csv"]
    selected_thresholds_path = output_metrics_dir / output_config["selected_thresholds_json"]
    test_metrics_path = output_metrics_dir / output_config["test_metrics_json"]

    val_predictions_df.to_csv(val_scores_path, index=False)
    test_thresholded_df.to_csv(test_scores_path, index=False)
    threshold_sweep_df.to_csv(threshold_sweep_path, index=False)

    selected_payload = {
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "label_names": label_names,
        "selected_thresholds": selected_thresholds,
    }

    selected_thresholds_path.write_text(
        json.dumps(selected_payload, indent=2),
        encoding="utf-8",
    )

    test_metrics_payload = {
        "label_names": label_names,
        "test_metrics": test_metrics,
    }

    test_metrics_path.write_text(
        json.dumps(test_metrics_payload, indent=2),
        encoding="utf-8",
    )

    coverage_risk_plot_path = output_figures_dir / output_config["coverage_risk_plot"]
    score_histogram_plot_path = output_figures_dir / output_config["score_histogram_plot"]

    plot_coverage_risk(
        sweep_df=threshold_sweep_df,
        output_path=coverage_risk_plot_path,
    )

    plot_score_histograms(
        test_predictions_df=test_predictions_df,
        selected_thresholds=selected_thresholds,
        output_path=score_histogram_plot_path,
    )

    docs_coverage_risk_path = docs_figures_dir / output_config["coverage_risk_plot"]
    docs_score_histogram_path = docs_figures_dir / output_config["score_histogram_plot"]

    shutil.copyfile(coverage_risk_plot_path, docs_coverage_risk_path)
    shutil.copyfile(score_histogram_plot_path, docs_score_histogram_path)

    markdown_report_path = docs_results_dir / output_config["markdown_report"]

    write_markdown_report(
        output_path=markdown_report_path,
        label_names=label_names,
        selected_thresholds=selected_thresholds,
        test_metrics=test_metrics,
        coverage_risk_plot_relative=f"figures/{output_config['coverage_risk_plot']}",
        score_histogram_relative=f"figures/{output_config['score_histogram_plot']}",
    )

    print("Open-set score reject baseline completed successfully.")
    print(f"Device: {device}")
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")

    print("\nSelected thresholds:")
    print(json.dumps(selected_thresholds, indent=2))

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))

    print("\nCreated files:")
    print(f"- validation scores: {val_scores_path}")
    print(f"- test scores: {test_scores_path}")
    print(f"- threshold sweep: {threshold_sweep_path}")
    print(f"- selected thresholds: {selected_thresholds_path}")
    print(f"- test metrics: {test_metrics_path}")
    print(f"- coverage-risk plot: {coverage_risk_plot_path}")
    print(f"- score histogram: {score_histogram_plot_path}")
    print(f"- thesis report: {markdown_report_path}")

    return {
        "selected_thresholds": selected_thresholds,
        "test_metrics": test_metrics,
        "markdown_report": str(markdown_report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run open-set score reject baseline."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/open_set_score_baseline_trashnet.yaml",
        help="Path to open-set score baseline YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    run_open_set_score_baseline(
        config_path=args.config,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()