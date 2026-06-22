from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from train_image_classifier import (
    WasteImageDataset,
    build_model,
    build_transforms,
    predict_dataset,
)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def add_active_learning_scores(
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []

    for row in rows:
        updated = dict(row)

        true_label = str(updated.get("true_label", ""))
        pred_label = str(updated.get("pred_label", ""))
        confidence = float(updated["confidence"])

        probabilities = [
            float(updated[f"prob_{class_name}"])
            for class_name in class_names
        ]
        sorted_probabilities = sorted(probabilities, reverse=True)

        top1_probability = sorted_probabilities[0]
        top2_probability = sorted_probabilities[1] if len(sorted_probabilities) > 1 else 0.0
        margin = top1_probability - top2_probability

        updated["prediction_matches_source_label"] = str(true_label == pred_label).lower()
        updated["uncertainty_score"] = 1.0 - confidence
        updated["probability_margin"] = margin

        if confidence < 0.50:
            updated["uncertainty_band"] = "very_high_uncertainty"
        elif confidence < 0.70:
            updated["uncertainty_band"] = "high_uncertainty"
        elif confidence < 0.85:
            updated["uncertainty_band"] = "medium_uncertainty"
        else:
            updated["uncertainty_band"] = "low_uncertainty"

        if true_label != pred_label and confidence >= 0.70:
            updated["selection_reason"] = "confident_wrong_candidate"
        elif confidence < 0.70:
            updated["selection_reason"] = "low_confidence_candidate"
        elif margin < 0.15:
            updated["selection_reason"] = "low_margin_candidate"
        else:
            updated["selection_reason"] = "not_priority"

        updated["human_review_status"] = "pending_review"
        updated["human_observation"] = ""
        updated["reviewed_label"] = ""
        updated["reviewed_action"] = ""
        updated["reviewer_notes"] = ""

        scored_rows.append(updated)

    scored_rows.sort(
        key=lambda item: (
            item["selection_reason"] == "not_priority",
            float(item["confidence"]),
            float(item["probability_margin"]),
        )
    )

    for rank, row in enumerate(scored_rows, start=1):
        row["active_learning_rank"] = rank

    return scored_rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)

    reason_counts: dict[str, int] = {}
    band_counts: dict[str, int] = {}
    true_label_counts: dict[str, int] = {}
    pred_label_counts: dict[str, int] = {}
    mismatch_count = 0

    for row in rows:
        reason = str(row["selection_reason"])
        band = str(row["uncertainty_band"])
        true_label = str(row["true_label"])
        pred_label = str(row["pred_label"])

        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        band_counts[band] = band_counts.get(band, 0) + 1
        true_label_counts[true_label] = true_label_counts.get(true_label, 0) + 1
        pred_label_counts[pred_label] = pred_label_counts.get(pred_label, 0) + 1

        if str(row["prediction_matches_source_label"]).lower() != "true":
            mismatch_count += 1

    return {
        "total_scored_candidates": total,
        "prediction_mismatch_count": mismatch_count,
        "prediction_mismatch_rate": mismatch_count / total if total else 0.0,
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "uncertainty_band_counts": dict(sorted(band_counts.items())),
        "true_label_counts": dict(sorted(true_label_counts.items())),
        "predicted_label_counts": dict(sorted(pred_label_counts.items())),
        "note": (
            "Selection is based mainly on uncertainty and model behaviour. "
            "Source labels are used as public-dataset reference labels, but selected samples "
            "still require human review before being added to active-learning training data."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score active-learning candidate pool using a trained OpenWaste-HR model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/configs/train_stage_01_trashnet.yaml"),
        help="Training config used to build the model.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ml/outputs/stage_01_trashnet_baseline/best_model.pt"),
        help="Trained model checkpoint path.",
    )

    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("ml/data/manifests/active_learning/stage_01_realwaste_candidate_pool_v1.csv"),
        help="Candidate pool CSV path.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ml/outputs/active_learning/stage_01_realwaste_candidate_scores_v1.csv"),
        help="Output scored candidate CSV path.",
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("ml/outputs/active_learning/stage_01_realwaste_candidate_scores_summary_v1.json"),
        help="Output summary JSON path.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = read_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, class_names = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
    )

    label_to_index = {label: index for index, label in enumerate(class_names)}

    eval_transform = build_transforms(
        Path(config["preprocessing"]["config_path"]),
        train=False,
    )

    dataset = WasteImageDataset(
        csv_path=args.candidate_csv,
        label_to_index=label_to_index,
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    result = predict_dataset(
        model=model,
        loader=loader,
        criterion=None,
        device=device,
        class_names=class_names,
        calculate_known_metrics=False,
    )

    scored_rows = add_active_learning_scores(
        rows=result["predictions"],
        class_names=class_names,
    )

    summary = build_summary(scored_rows)

    save_csv(args.output_csv, scored_rows)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Scored candidates written: {args.output_csv}")
    print(f"Summary written: {args.summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
