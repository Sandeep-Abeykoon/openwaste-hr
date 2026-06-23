from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_reject_option_open_set import (
    choose_threshold,
    compute_auroc,
    evaluate_threshold,
    expected_calibration_error,
    load_model_from_checkpoint,
    predict_split,
    read_yaml,
    write_csv,
)
from train_image_classifier import WasteImageDataset, build_transforms


SCORE_COLUMN = "temperature_scaled_confidence"


def collect_logits(
    *,
    csv_path: Path,
    model: nn.Module,
    label_to_index: dict[str, int],
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    dataset = WasteImageDataset(
        csv_path=csv_path,
        label_to_index=label_to_index,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_logits: list[torch.Tensor] = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            all_logits.append(logits.detach().cpu())

    return torch.cat(all_logits, dim=0)


def labels_from_predictions(
    *,
    rows: list[dict[str, Any]],
    label_to_index: dict[str, int],
) -> torch.Tensor:
    labels: list[int] = []

    for row in rows:
        true_label = str(row.get("true_label", "")).strip()

        if true_label not in label_to_index:
            raise ValueError(f"Cannot temperature-scale unknown/non-known label: {true_label}")

        labels.append(label_to_index[true_label])

    return torch.tensor(labels, dtype=torch.long)


def fit_temperature(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    logits = logits.to(device)
    labels = labels.to(device)

    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.05,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()

        temperature = torch.exp(log_temperature).clamp(min=0.05, max=10.0)
        loss = F.cross_entropy(logits / temperature, labels)

        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        temperature = torch.exp(log_temperature).clamp(min=0.05, max=10.0)

    return float(temperature.item())


def add_temperature_scaled_confidence(
    *,
    rows: list[dict[str, Any]],
    logits: torch.Tensor,
    temperature: float,
) -> list[dict[str, Any]]:
    scaled_probs = F.softmax(logits / temperature, dim=1)
    confidences, _ = torch.max(scaled_probs, dim=1)

    updated_rows: list[dict[str, Any]] = []

    for row, confidence in zip(rows, confidences.tolist()):
        updated = dict(row)
        updated[SCORE_COLUMN] = float(confidence)
        updated_rows.append(updated)

    return updated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate temperature-scaled confidence for reject-option/open-set evaluation."
    )

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = read_yaml(args.config)
    data_config = config["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model, class_names = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
    )

    label_to_index = {label: index for index, label in enumerate(class_names)}

    transform = build_transforms(
        Path(config["preprocessing"]["config_path"]),
        train=False,
    )

    known_val = predict_split(
        split_name="known_val",
        csv_path=Path(data_config["val_csv"]),
        model=model,
        class_names=class_names,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calculate_known_metrics=True,
    )

    known_test = predict_split(
        split_name="known_test",
        csv_path=Path(data_config["test_csv"]),
        model=model,
        class_names=class_names,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calculate_known_metrics=True,
    )

    unknown_val = predict_split(
        split_name="unknown_val",
        csv_path=Path(data_config["unknown_val_csv"]),
        model=model,
        class_names=class_names,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calculate_known_metrics=False,
    )

    unknown_test = predict_split(
        split_name="unknown_test",
        csv_path=Path(data_config["unknown_test_csv"]),
        model=model,
        class_names=class_names,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        calculate_known_metrics=False,
    )

    known_val_logits = collect_logits(
        csv_path=Path(data_config["val_csv"]),
        model=model,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    known_test_logits = collect_logits(
        csv_path=Path(data_config["test_csv"]),
        model=model,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unknown_val_logits = collect_logits(
        csv_path=Path(data_config["unknown_val_csv"]),
        model=model,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unknown_test_logits = collect_logits(
        csv_path=Path(data_config["unknown_test_csv"]),
        model=model,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    known_val_labels = labels_from_predictions(
        rows=known_val,
        label_to_index=label_to_index,
    )

    temperature = fit_temperature(
        logits=known_val_logits,
        labels=known_val_labels,
        device=device,
    )

    known_val_scaled = add_temperature_scaled_confidence(
        rows=known_val,
        logits=known_val_logits,
        temperature=temperature,
    )

    known_test_scaled = add_temperature_scaled_confidence(
        rows=known_test,
        logits=known_test_logits,
        temperature=temperature,
    )

    unknown_val_scaled = add_temperature_scaled_confidence(
        rows=unknown_val,
        logits=unknown_val_logits,
        temperature=temperature,
    )

    unknown_test_scaled = add_temperature_scaled_confidence(
        rows=unknown_test,
        logits=unknown_test_logits,
        temperature=temperature,
    )

    selected_val = choose_threshold(
        known_rows=known_val_scaled,
        unknown_rows=unknown_val_scaled,
        score_column=SCORE_COLUMN,
        higher_is_known=True,
    )

    val_auroc = compute_auroc(
        known_rows=known_val_scaled,
        unknown_rows=unknown_val_scaled,
        score_column=SCORE_COLUMN,
        higher_is_known=True,
    )

    test_result = evaluate_threshold(
        known_rows=known_test_scaled,
        unknown_rows=unknown_test_scaled,
        score_column=SCORE_COLUMN,
        threshold=selected_val["threshold"],
        higher_is_known=True,
    )

    test_auroc = compute_auroc(
        known_rows=known_test_scaled,
        unknown_rows=unknown_test_scaled,
        score_column=SCORE_COLUMN,
        higher_is_known=True,
    )

    metric_row = {
        "score_method": SCORE_COLUMN,
        "temperature": temperature,
        "direction": "higher_is_known",
        "selected_threshold_from_val": selected_val["threshold"],
        "val_binary_balanced_accuracy": selected_val["binary_balanced_accuracy"],
        "val_known_coverage": selected_val["known_coverage"],
        "val_unknown_rejection_rate": selected_val["unknown_rejection_rate"],
        "val_known_accuracy_accepted": selected_val["known_accuracy_accepted"],
        "val_selective_risk": selected_val["selective_risk"],
        "val_auroc_known_vs_unknown": val_auroc,
        "test_binary_balanced_accuracy": test_result["binary_balanced_accuracy"],
        "test_known_coverage": test_result["known_coverage"],
        "test_unknown_rejection_rate": test_result["unknown_rejection_rate"],
        "test_known_accuracy_accepted": test_result["known_accuracy_accepted"],
        "test_selective_risk": test_result["selective_risk"],
        "test_auroc_known_vs_unknown": test_auroc,
    }

    summary = {
        "experiment": config["experiment"]["name"],
        "checkpoint": str(args.checkpoint),
        "temperature": temperature,
        "known_val_rows": len(known_val_scaled),
        "known_test_rows": len(known_test_scaled),
        "unknown_val_rows": len(unknown_val_scaled),
        "unknown_test_rows": len(unknown_test_scaled),
        "known_val_ece_before_temperature": expected_calibration_error(
            rows=known_val,
            confidence_column="confidence",
        ),
        "known_val_ece_after_temperature": expected_calibration_error(
            rows=known_val_scaled,
            confidence_column=SCORE_COLUMN,
        ),
        "known_test_ece_before_temperature": expected_calibration_error(
            rows=known_test,
            confidence_column="confidence",
        ),
        "known_test_ece_after_temperature": expected_calibration_error(
            rows=known_test_scaled,
            confidence_column=SCORE_COLUMN,
        ),
        "method": metric_row,
        "rule": (
            "Temperature is fitted only on known_val labels. "
            "Reject threshold is selected using known_val + unknown_val. "
            "Final performance is reported on known_test + unknown_test."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "temperature_scaled_summary_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_csv(args.output_dir / "temperature_scaled_metrics_v1.csv", [metric_row])
    write_csv(
        args.output_dir / "temperature_scaled_predictions_v1.csv",
        known_val_scaled + known_test_scaled + unknown_val_scaled + unknown_test_scaled,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
