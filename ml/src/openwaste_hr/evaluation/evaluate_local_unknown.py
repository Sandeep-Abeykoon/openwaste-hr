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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from openwaste_hr.data.manifest import load_manifest
from openwaste_hr.evaluation.open_set_scores import (
    compute_energy_score,
    compute_max_logit_score,
)
from openwaste_hr.evaluation.unknown_metrics import (
    apply_unknown_decision_rule,
    build_accepted_label_distribution,
    compute_unknown_rejection_metrics,
)
from openwaste_hr.models.baseline_cnn import create_baseline_cnn
from openwaste_hr.training.image_transforms import build_eval_transform


class LocalUnknownImageDataset(Dataset):
    """
    Dataset for local unknown images.

    The labels are intentionally unknown, so this dataset does not encode them
    into known class IDs.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        transform=None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.transform = transform

        manifest = load_manifest(self.manifest_path)

        if manifest.empty:
            raise ValueError("Local unknown manifest is empty.")

        self.manifest = manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.manifest):
            raise IndexError(f"Index out of range: {index}")

        row = self.manifest.iloc[index]
        image_path = self.project_root / str(row["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "sample_id": str(row["sample_id"]),
            "image_path": str(row["image_path"]),
            "source_dataset": str(row["source_dataset"]),
            "source_split": str(row["source_split"]),
            "original_label": str(row["original_label"]),
            "fine_label": str(row["fine_label"]),
            "coarse_label": str(row["coarse_label"]),
            "usage": str(row["usage"]),
            "object_description": str(row.get("object_description", "")),
            "why_unknown_or_difficult": str(row.get("why_unknown_or_difficult", "")),
            "lighting_condition": str(row.get("lighting_condition", "")),
            "background_condition": str(row.get("background_condition", "")),
        }


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML dictionary.")

    return config


def resolve_path(project_root: str | Path, path_text: str | Path) -> Path:
    return Path(project_root) / Path(path_text)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def load_json(json_path: str | Path) -> dict[str, Any]:
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    return json.loads(json_path.read_text(encoding="utf-8"))


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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


def build_unknown_loader(
    manifest_path: Path,
    project_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = LocalUnknownImageDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        transform=build_eval_transform(image_size=image_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


@torch.no_grad()
def predict_local_unknown(
    model,
    dataloader: DataLoader,
    device: torch.device,
    label_names: list[str],
    energy_temperature: float,
) -> pd.DataFrame:
    model.eval()

    rows: list[dict[str, Any]] = []

    for batch in dataloader:
        images = batch["image"].to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        max_probs, predictions = torch.max(probabilities, dim=1)

        max_logit_scores = compute_max_logit_score(logits)
        energy_scores = compute_energy_score(
            logits=logits,
            temperature=energy_temperature,
        )

        batch_size = images.size(0)

        for index in range(batch_size):
            pred_id = int(predictions[index].detach().cpu().item())
            pred_label = label_names[pred_id]

            row = {
                "sample_id": str(batch["sample_id"][index]),
                "image_path": str(batch["image_path"][index]),
                "source_dataset": str(batch["source_dataset"][index]),
                "source_split": str(batch["source_split"][index]),
                "original_label": str(batch["original_label"][index]),
                "fine_label": str(batch["fine_label"][index]),
                "coarse_label": str(batch["coarse_label"][index]),
                "usage": str(batch["usage"][index]),
                "object_description": str(batch["object_description"][index]),
                "why_unknown_or_difficult": str(batch["why_unknown_or_difficult"][index]),
                "lighting_condition": str(batch["lighting_condition"][index]),
                "background_condition": str(batch["background_condition"][index]),
                "pred_label_id": pred_id,
                "pred_label": pred_label,
                "max_softmax_confidence": round(
                    float(max_probs[index].detach().cpu().item()),
                    6,
                ),
                "max_logit_score": round(
                    float(max_logit_scores[index].detach().cpu().item()),
                    6,
                ),
                "energy_score": round(
                    float(energy_scores[index].detach().cpu().item()),
                    6,
                ),
            }

            for label_index, label_name in enumerate(label_names):
                row[f"prob_{label_name}"] = round(
                    float(probabilities[index, label_index].detach().cpu().item()),
                    6,
                )

            rows.append(row)

    return pd.DataFrame(rows)


def extract_confidence_threshold(payload: dict[str, Any]) -> float:
    return float(payload["selected_threshold"]["threshold"])


def extract_open_set_thresholds(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return payload["selected_thresholds"]


def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
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


def plot_rejection_rates(metrics_by_method: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    method_names = list(metrics_by_method.keys())
    rejection_rates = [
        metrics_by_method[method]["unknown_rejection_rate"]
        for method in method_names
    ]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.bar(method_names, rejection_rates)
    axis.set_title("Local Unknown Rejection Rate by Method")
    axis.set_xlabel("Method")
    axis.set_ylabel("Unknown Rejection Rate")
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(axis="x", rotation=25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_accepted_label_distribution(
    accepted_distribution_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if accepted_distribution_df.empty:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.text(
            0.5,
            0.5,
            "No unknown images were accepted as known",
            ha="center",
            va="center",
        )
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        return

    pivot = accepted_distribution_df.pivot_table(
        index="pred_label",
        columns="method_name",
        values="accepted_count",
        fill_value=0,
        aggfunc="sum",
    )

    figure, axis = plt.subplots(figsize=(9, 6))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Accepted Unknown Images by Predicted Label")
    axis.set_xlabel("Predicted Known Label")
    axis.set_ylabel("Accepted Unknown Count")
    axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_markdown_report(
    output_path: Path,
    label_names: list[str],
    thresholds_summary_df: pd.DataFrame,
    metrics_summary_df: pd.DataFrame,
    accepted_distribution_df: pd.DataFrame,
    rejection_plot_relative: str,
    accepted_label_plot_relative: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = f"""# Local Unknown Evaluation v1

## Purpose

This report evaluates whether the current OpenWaste-HR reject baselines can route local phone-captured unknown images to manual review.

## Known Labels Used by the Model

{", ".join(label_names)}

## Thresholds Used

Thresholds were selected from validation data in earlier experiments. The local unknown set was not used for threshold selection.

{dataframe_to_markdown_table(thresholds_summary_df)}

## Unknown Rejection Metrics

{dataframe_to_markdown_table(metrics_summary_df)}

## Accepted Unknown Label Distribution

{dataframe_to_markdown_table(accepted_distribution_df)}

## Rejection Rate Plot

![Local unknown rejection rates]({rejection_plot_relative})

## Accepted Label Distribution Plot

![Accepted unknown label distribution]({accepted_label_plot_relative})

## Research Interpretation

For unknown evaluation, rejection/manual review is the desired behaviour.

Accepted unknown samples are treated as false accepts because the system forced a local unknown item into a known fine label. This result is important because it tests the actual OpenWaste-HR motivation: avoiding unsafe forced predictions on unknown or locally shifted inputs.
"""

    output_path.write_text(report, encoding="utf-8")


def run_local_unknown_evaluation(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    data_config = config["data"]
    score_config = config["score_methods"]
    output_config = config["outputs"]

    checkpoint_path = resolve_path(project_root, paths["checkpoint"])
    local_unknown_manifest_path = resolve_path(project_root, paths["local_unknown_manifest"])
    confidence_threshold_path = resolve_path(project_root, paths["confidence_threshold_json"])
    open_set_thresholds_path = resolve_path(project_root, paths["open_set_thresholds_json"])

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

    loader = build_unknown_loader(
        manifest_path=local_unknown_manifest_path,
        project_root=project_root,
        image_size=int(data_config["image_size"]),
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
    )

    predictions_df = predict_local_unknown(
        model=model,
        dataloader=loader,
        device=device,
        label_names=label_names,
        energy_temperature=float(score_config["energy"].get("temperature", 1.0)),
    )

    confidence_threshold_payload = load_json(confidence_threshold_path)
    open_set_thresholds_payload = load_json(open_set_thresholds_path)

    confidence_threshold = extract_confidence_threshold(confidence_threshold_payload)
    open_set_thresholds = extract_open_set_thresholds(open_set_thresholds_payload)

    method_rules = []

    if bool(score_config["confidence"]["enabled"]):
        method_rules.append(
            {
                "method_name": "confidence",
                "score_column": "max_softmax_confidence",
                "threshold": confidence_threshold,
                "accept_direction": "greater_equal",
            }
        )

    if bool(score_config["max_logit"]["enabled"]):
        method_rules.append(
            {
                "method_name": "max_logit",
                "score_column": "max_logit_score",
                "threshold": float(open_set_thresholds["max_logit"]["threshold"]),
                "accept_direction": str(open_set_thresholds["max_logit"]["accept_direction"]),
            }
        )

    if bool(score_config["energy"]["enabled"]):
        method_rules.append(
            {
                "method_name": "energy",
                "score_column": "energy_score",
                "threshold": float(open_set_thresholds["energy"]["threshold"]),
                "accept_direction": str(open_set_thresholds["energy"]["accept_direction"]),
            }
        )

    all_decision_parts = []
    metrics_by_method: dict[str, Any] = {}

    for rule in method_rules:
        decisions_df = apply_unknown_decision_rule(
            predictions_df=predictions_df,
            method_name=rule["method_name"],
            score_column=rule["score_column"],
            threshold=float(rule["threshold"]),
            accept_direction=rule["accept_direction"],
        )

        metrics = compute_unknown_rejection_metrics(decisions_df)
        metrics_by_method[rule["method_name"]] = metrics
        all_decision_parts.append(decisions_df)

    method_decisions_df = pd.concat(all_decision_parts, ignore_index=True)

    accepted_distribution_df = build_accepted_label_distribution(method_decisions_df)

    thresholds_summary_df = pd.DataFrame(method_rules)

    metrics_summary_df = pd.DataFrame(
        [
            {
                "method_name": method_name,
                **metrics,
            }
            for method_name, metrics in metrics_by_method.items()
        ]
    )

    predictions_path = output_metrics_dir / output_config["predictions_csv"]
    method_decisions_path = output_metrics_dir / output_config["method_decisions_csv"]
    metrics_json_path = output_metrics_dir / output_config["metrics_json"]
    accepted_distribution_path = output_metrics_dir / output_config["accepted_label_distribution_csv"]

    predictions_df.to_csv(predictions_path, index=False)
    method_decisions_df.to_csv(method_decisions_path, index=False)
    accepted_distribution_df.to_csv(accepted_distribution_path, index=False)

    metrics_payload = {
        "label_names": label_names,
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "thresholds": method_rules,
        "metrics_by_method": metrics_by_method,
    }

    metrics_json_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    rejection_rate_plot_path = output_figures_dir / output_config["rejection_rate_plot"]
    accepted_label_plot_path = output_figures_dir / output_config["accepted_label_plot"]

    plot_rejection_rates(
        metrics_by_method=metrics_by_method,
        output_path=rejection_rate_plot_path,
    )

    plot_accepted_label_distribution(
        accepted_distribution_df=accepted_distribution_df,
        output_path=accepted_label_plot_path,
    )

    docs_rejection_rate_plot_path = docs_figures_dir / output_config["rejection_rate_plot"]
    docs_accepted_label_plot_path = docs_figures_dir / output_config["accepted_label_plot"]

    shutil.copyfile(rejection_rate_plot_path, docs_rejection_rate_plot_path)
    shutil.copyfile(accepted_label_plot_path, docs_accepted_label_plot_path)

    markdown_report_path = docs_results_dir / output_config["markdown_report"]

    write_markdown_report(
        output_path=markdown_report_path,
        label_names=label_names,
        thresholds_summary_df=thresholds_summary_df,
        metrics_summary_df=metrics_summary_df,
        accepted_distribution_df=accepted_distribution_df,
        rejection_plot_relative=f"figures/{output_config['rejection_rate_plot']}",
        accepted_label_plot_relative=f"figures/{output_config['accepted_label_plot']}",
    )

    print("Local unknown evaluation completed successfully.")
    print(f"Device: {device}")
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Unknown samples: {len(predictions_df)}")
    print("\nMetrics by method:")
    print(json.dumps(metrics_by_method, indent=2))
    print("\nCreated files:")
    print(f"- predictions: {predictions_path}")
    print(f"- method decisions: {method_decisions_path}")
    print(f"- metrics: {metrics_json_path}")
    print(f"- accepted label distribution: {accepted_distribution_path}")
    print(f"- rejection-rate plot: {rejection_rate_plot_path}")
    print(f"- accepted-label plot: {accepted_label_plot_path}")
    print(f"- thesis report: {markdown_report_path}")

    return {
        "metrics_by_method": metrics_by_method,
        "markdown_report": str(markdown_report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OpenWaste-HR on local unknown images."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/local_unknown_evaluation.yaml",
        help="Path to local unknown evaluation YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    run_local_unknown_evaluation(
        config_path=args.config,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()