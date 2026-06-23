from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from train_image_classifier import (
    WasteImageDataset,
    build_model,
    build_transforms,
    predict_dataset,
)


SCORE_COLUMNS = [
    "confidence",
    "max_logit",
    "energy",
]


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = list(checkpoint.get("class_names", config["labels"]["known_classes"]))

    model = build_model(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def predict_split(
    *,
    split_name: str,
    csv_path: Path,
    model: nn.Module,
    class_names: list[str],
    label_to_index: dict[str, int],
    transform: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    calculate_known_metrics: bool,
) -> list[dict[str, Any]]:
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

    result = predict_dataset(
        model=model,
        loader=loader,
        criterion=None,
        device=device,
        class_names=class_names,
        calculate_known_metrics=calculate_known_metrics,
    )

    predictions = result["predictions"]

    for row in predictions:
        row["split"] = split_name

    return predictions


def as_float(row: dict[str, Any], column: str) -> float:
    value = row.get(column, "")

    if value == "":
        raise ValueError(f"Missing score value for column: {column}")

    return float(value)


def is_correct(row: dict[str, Any]) -> bool:
    return str(row.get("true_label", "")).strip() == str(row.get("pred_label", "")).strip()


def expected_calibration_error(
    *,
    rows: list[dict[str, Any]],
    confidence_column: str = "confidence",
    n_bins: int = 15,
) -> float:
    if not rows:
        return 0.0

    confidences = np.array([as_float(row, confidence_column) for row in rows], dtype=float)
    correct = np.array([1.0 if is_correct(row) else 0.0 for row in rows], dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for bin_index in range(n_bins):
        low = bin_edges[bin_index]
        high = bin_edges[bin_index + 1]

        if bin_index == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        if not np.any(mask):
            continue

        bin_confidence = float(np.mean(confidences[mask]))
        bin_accuracy = float(np.mean(correct[mask]))
        bin_weight = float(np.mean(mask))

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def infer_score_direction(
    *,
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    score_column: str,
) -> bool:
    known_mean = float(np.mean([as_float(row, score_column) for row in known_rows]))
    unknown_mean = float(np.mean([as_float(row, score_column) for row in unknown_rows]))

    return known_mean >= unknown_mean


def accepted_as_known(score: float, threshold: float, higher_is_known: bool) -> bool:
    if higher_is_known:
        return score >= threshold

    return score <= threshold


def evaluate_threshold(
    *,
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    score_column: str,
    threshold: float,
    higher_is_known: bool,
) -> dict[str, Any]:
    known_accept_flags = [
        accepted_as_known(
            score=as_float(row, score_column),
            threshold=threshold,
            higher_is_known=higher_is_known,
        )
        for row in known_rows
    ]

    unknown_accept_flags = [
        accepted_as_known(
            score=as_float(row, score_column),
            threshold=threshold,
            higher_is_known=higher_is_known,
        )
        for row in unknown_rows
    ]

    accepted_known_rows = [
        row for row, accepted in zip(known_rows, known_accept_flags)
        if accepted
    ]

    known_coverage = (
        sum(known_accept_flags) / len(known_accept_flags)
        if known_accept_flags else 0.0
    )

    known_rejection_rate = 1.0 - known_coverage

    unknown_acceptance_rate = (
        sum(unknown_accept_flags) / len(unknown_accept_flags)
        if unknown_accept_flags else 0.0
    )

    unknown_rejection_rate = 1.0 - unknown_acceptance_rate

    known_accuracy_all = (
        sum(1 for row in known_rows if is_correct(row)) / len(known_rows)
        if known_rows else 0.0
    )

    known_accuracy_accepted = (
        sum(1 for row in accepted_known_rows if is_correct(row)) / len(accepted_known_rows)
        if accepted_known_rows else 0.0
    )

    selective_risk = 1.0 - known_accuracy_accepted if accepted_known_rows else 1.0

    binary_balanced_accuracy = (known_coverage + unknown_rejection_rate) / 2.0

    return {
        "threshold": float(threshold),
        "higher_is_known": higher_is_known,
        "known_rows": len(known_rows),
        "unknown_rows": len(unknown_rows),
        "known_coverage": float(known_coverage),
        "known_rejection_rate": float(known_rejection_rate),
        "unknown_rejection_rate": float(unknown_rejection_rate),
        "unknown_acceptance_rate": float(unknown_acceptance_rate),
        "known_accuracy_all": float(known_accuracy_all),
        "known_accuracy_accepted": float(known_accuracy_accepted),
        "selective_risk": float(selective_risk),
        "binary_balanced_accuracy": float(binary_balanced_accuracy),
    }


def choose_threshold(
    *,
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    score_column: str,
    higher_is_known: bool,
) -> dict[str, Any]:
    scores = np.array(
        [as_float(row, score_column) for row in known_rows + unknown_rows],
        dtype=float,
    )

    unique_scores = np.unique(scores)

    if len(unique_scores) > 500:
        thresholds = np.quantile(unique_scores, np.linspace(0.0, 1.0, 500))
    else:
        thresholds = unique_scores

    best_result: dict[str, Any] | None = None

    for threshold in thresholds:
        result = evaluate_threshold(
            known_rows=known_rows,
            unknown_rows=unknown_rows,
            score_column=score_column,
            threshold=float(threshold),
            higher_is_known=higher_is_known,
        )

        if best_result is None:
            best_result = result
            continue

        current_key = (
            result["binary_balanced_accuracy"],
            result["known_accuracy_accepted"],
            result["known_coverage"],
            result["unknown_rejection_rate"],
        )

        best_key = (
            best_result["binary_balanced_accuracy"],
            best_result["known_accuracy_accepted"],
            best_result["known_coverage"],
            best_result["unknown_rejection_rate"],
        )

        if current_key > best_key:
            best_result = result

    if best_result is None:
        raise ValueError("Could not select threshold.")

    return best_result


def compute_auroc(
    *,
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    score_column: str,
    higher_is_known: bool,
) -> float:
    y_true = np.array([1] * len(known_rows) + [0] * len(unknown_rows), dtype=int)

    scores = np.array(
        [as_float(row, score_column) for row in known_rows + unknown_rows],
        dtype=float,
    )

    if not higher_is_known:
        scores = -scores

    return float(roc_auc_score(y_true, scores))


def build_coverage_risk_rows(
    *,
    split_name: str,
    score_column: str,
    known_rows: list[dict[str, Any]],
    higher_is_known: bool,
) -> list[dict[str, Any]]:
    scores = np.array([as_float(row, score_column) for row in known_rows], dtype=float)

    if len(scores) == 0:
        return []

    thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 101))

    output_rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        accepted_rows = [
            row for row in known_rows
            if accepted_as_known(
                score=as_float(row, score_column),
                threshold=float(threshold),
                higher_is_known=higher_is_known,
            )
        ]

        coverage = len(accepted_rows) / len(known_rows)

        accuracy_accepted = (
            sum(1 for row in accepted_rows if is_correct(row)) / len(accepted_rows)
            if accepted_rows else 0.0
        )

        risk = 1.0 - accuracy_accepted if accepted_rows else 1.0

        output_rows.append(
            {
                "split": split_name,
                "score_method": score_column,
                "threshold": float(threshold),
                "coverage": float(coverage),
                "risk": float(risk),
                "accepted_known_rows": len(accepted_rows),
                "total_known_rows": len(known_rows),
            }
        )

    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reject-option/open-set scoring for OpenWaste-HR."
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

    all_predictions = known_val + known_test + unknown_val + unknown_test

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "open_set_predictions_v1.csv", all_predictions)

    metric_rows: list[dict[str, Any]] = []
    coverage_risk_rows: list[dict[str, Any]] = []

    for score_column in SCORE_COLUMNS:
        higher_is_known = infer_score_direction(
            known_rows=known_val,
            unknown_rows=unknown_val,
            score_column=score_column,
        )

        selected_val = choose_threshold(
            known_rows=known_val,
            unknown_rows=unknown_val,
            score_column=score_column,
            higher_is_known=higher_is_known,
        )

        val_auroc = compute_auroc(
            known_rows=known_val,
            unknown_rows=unknown_val,
            score_column=score_column,
            higher_is_known=higher_is_known,
        )

        test_result = evaluate_threshold(
            known_rows=known_test,
            unknown_rows=unknown_test,
            score_column=score_column,
            threshold=selected_val["threshold"],
            higher_is_known=higher_is_known,
        )

        test_auroc = compute_auroc(
            known_rows=known_test,
            unknown_rows=unknown_test,
            score_column=score_column,
            higher_is_known=higher_is_known,
        )

        metric_rows.append(
            {
                "score_method": score_column,
                "direction": "higher_is_known" if higher_is_known else "lower_is_known",
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
        )

        coverage_risk_rows.extend(
            build_coverage_risk_rows(
                split_name="known_val",
                score_column=score_column,
                known_rows=known_val,
                higher_is_known=higher_is_known,
            )
        )

        coverage_risk_rows.extend(
            build_coverage_risk_rows(
                split_name="known_test",
                score_column=score_column,
                known_rows=known_test,
                higher_is_known=higher_is_known,
            )
        )

    summary = {
        "experiment": config["experiment"]["name"],
        "checkpoint": str(args.checkpoint),
        "known_val_rows": len(known_val),
        "known_test_rows": len(known_test),
        "unknown_val_rows": len(unknown_val),
        "unknown_test_rows": len(unknown_test),
        "unknown_labels": sorted(set(row.get("true_label", "") for row in unknown_test)),
        "known_val_ece": expected_calibration_error(rows=known_val),
        "known_test_ece": expected_calibration_error(rows=known_test),
        "methods": metric_rows,
        "rule": (
            "Thresholds are selected using known_val + unknown_val only. "
            "Final open-set performance is reported on known_test + unknown_test."
        ),
    }

    (args.output_dir / "open_set_summary_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_csv(args.output_dir / "open_set_metrics_v1.csv", metric_rows)
    write_csv(args.output_dir / "coverage_risk_curve_v1.csv", coverage_risk_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
