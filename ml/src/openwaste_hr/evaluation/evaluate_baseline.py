from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from openwaste_hr.evaluation.classification_metrics import (
    build_classification_report_dataframe,
    build_confusion_matrix_dataframe,
    build_prediction_dataframe,
    compute_summary_metrics,
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
    Resolve evaluation device.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def dataframe_to_markdown_table(dataframe) -> str:
    """
    Convert a DataFrame into a Markdown table without requiring tabulate.
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


def plot_confusion_matrix(
    confusion_matrix_df,
    label_names: list[str],
    output_path: Path,
) -> None:
    """
    Save a confusion matrix plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion_matrix_df.values)

    axis.set_title("Baseline TrashNet Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(range(len(label_names)))
    axis.set_yticks(range(len(label_names)))
    axis.set_xticklabels(label_names, rotation=35, ha="right")
    axis.set_yticklabels(label_names)

    for row_index in range(len(label_names)):
        for col_index in range(len(label_names)):
            value = int(confusion_matrix_df.values[row_index, col_index])
            axis.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_markdown_report(
    output_path: Path,
    checkpoint_path: Path,
    label_names: list[str],
    summary_metrics: dict[str, float],
    classification_report_df,
    confusion_matrix_plot_relative: str,
) -> None:
    """
    Write thesis-ready baseline report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_rows = [
        {"metric": key, "value": round(value, 6)}
        for key, value in summary_metrics.items()
    ]

    import pandas as pd

    main_metrics_df = pd.DataFrame(main_rows)

    report = f"""# Baseline TrashNet Evaluation v1

## Model

| Item | Value |
|---|---|
| Checkpoint | `{checkpoint_path.as_posix()}` |
| Model type | Closed-set forced-choice classifier |
| Dataset | TrashNet mapped into OpenWaste-HR taxonomy |
| Label level | Fine label |
| Number of classes | {len(label_names)} |
| Classes | {", ".join(label_names)} |

## Main Test Metrics

{dataframe_to_markdown_table(main_metrics_df)}

## Classification Report

{dataframe_to_markdown_table(classification_report_df.round(6))}

## Confusion Matrix

![Baseline confusion matrix]({confusion_matrix_plot_relative})

## Research Interpretation

This is the first closed-set baseline. It always predicts one of the known TrashNet-derived fine labels.

This result should not be treated as the final OpenWaste-HR contribution. It is the comparison point for later experiments involving confidence-based rejection, unknown detection, hierarchical coarse fallback, and local active learning.

Important limitation: this TrashNet baseline does not include organic or e-waste/battery classes, and it does not test unknown rejection.
"""

    output_path.write_text(report, encoding="utf-8")


@torch.no_grad()
def run_prediction_loop(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[int], list[int], list[float], list[list[float]]]:
    """
    Run model over the test set and collect predictions.
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

    return metadata_rows, y_true, y_pred, confidences, probability_rows


def evaluate_baseline(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """
    Evaluate the saved baseline checkpoint on the test set.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    data_config = config["data"]
    output_config = config["outputs"]

    checkpoint_path = resolve_path(project_root, paths["checkpoint"])
    test_manifest_path = resolve_path(project_root, paths["test_manifest"])
    output_metrics_dir = resolve_path(project_root, paths["output_metrics_dir"])
    output_figures_dir = resolve_path(project_root, paths["output_figures_dir"])
    docs_results_dir = resolve_path(project_root, paths["docs_results_dir"])
    docs_figures_dir = resolve_path(project_root, paths["docs_figures_dir"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run baseline training before evaluation."
        )

    device = get_device(str(data_config["device"]))

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    label_names = checkpoint["label_names"]
    label_to_id = build_label_to_id(label_names)
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

    test_dataset = TorchManifestImageDataset(
        manifest_path=test_manifest_path,
        project_root=project_root,
        label_to_id=label_to_id,
        usage_filter=["known_test"],
        label_column=str(data_config["label_column"]),
        transform=build_eval_transform(image_size=int(data_config["image_size"])),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
    )

    metadata_rows, y_true, y_pred, confidences, probability_rows = run_prediction_loop(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    summary_metrics = compute_summary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    predictions_df = build_prediction_dataframe(
        metadata_rows=metadata_rows,
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        probability_rows=probability_rows,
        label_names=label_names,
    )

    confusion_matrix_df = build_confusion_matrix_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    classification_report_df = build_classification_report_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    docs_results_dir.mkdir(parents=True, exist_ok=True)
    docs_figures_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_metrics_dir / output_config["predictions_csv"]
    metrics_json_path = output_metrics_dir / output_config["summary_metrics_json"]
    confusion_matrix_csv_path = output_metrics_dir / output_config["confusion_matrix_csv"]
    classification_report_csv_path = output_metrics_dir / output_config["classification_report_csv"]
    confusion_matrix_plot_path = output_figures_dir / output_config["confusion_matrix_plot"]

    predictions_df.to_csv(predictions_path, index=False)
    confusion_matrix_df.to_csv(confusion_matrix_csv_path)
    classification_report_df.to_csv(classification_report_csv_path, index=False)

    metrics_payload = {
        "label_names": label_names,
        "test_sample_count": len(y_true),
        "metrics": summary_metrics,
    }

    metrics_json_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    plot_confusion_matrix(
        confusion_matrix_df=confusion_matrix_df,
        label_names=label_names,
        output_path=confusion_matrix_plot_path,
    )

    docs_confusion_matrix_path = docs_figures_dir / output_config["confusion_matrix_plot"]
    shutil.copyfile(confusion_matrix_plot_path, docs_confusion_matrix_path)

    markdown_report_path = docs_results_dir / output_config["markdown_report"]

    write_markdown_report(
        output_path=markdown_report_path,
        checkpoint_path=checkpoint_path,
        label_names=label_names,
        summary_metrics=summary_metrics,
        classification_report_df=classification_report_df,
        confusion_matrix_plot_relative=f"figures/{output_config['confusion_matrix_plot']}",
    )

    print("Baseline evaluation completed successfully.")
    print(f"Device: {device}")
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy: {summary_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {summary_metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {summary_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {summary_metrics['weighted_f1']:.4f}")
    print("\nCreated files:")
    print(f"- predictions: {predictions_path}")
    print(f"- metrics: {metrics_json_path}")
    print(f"- confusion matrix CSV: {confusion_matrix_csv_path}")
    print(f"- classification report: {classification_report_csv_path}")
    print(f"- confusion matrix plot: {confusion_matrix_plot_path}")
    print(f"- thesis report: {markdown_report_path}")

    return {
        "metrics": summary_metrics,
        "predictions_csv": str(predictions_path),
        "metrics_json": str(metrics_json_path),
        "markdown_report": str(markdown_report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OpenWaste-HR baseline checkpoint."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/evaluate_baseline_trashnet.yaml",
        help="Path to evaluation YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    evaluate_baseline(
        config_path=args.config,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()