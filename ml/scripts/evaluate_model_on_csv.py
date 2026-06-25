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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


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


def save_temp_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows available after filtering.")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OpenWaste-HR model on any manifest CSV."
    )

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--source-dataset-filter", type=str, default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = read_yaml(args.config)

    rows = read_csv(args.csv)

    if args.source_dataset_filter:
        rows = [
            row for row in rows
            if row.get("source_dataset", "") == args.source_dataset_filter
        ]

    if not rows:
        raise ValueError("No rows found for evaluation after filtering.")

    temp_csv = args.output_json.parent / "_temp_eval_manifest.csv"
    save_temp_csv(temp_csv, rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluation rows: {len(rows)}")

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
        csv_path=temp_csv,
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
        calculate_known_metrics=True,
    )

    output = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "input_csv": str(args.csv),
        "source_dataset_filter": args.source_dataset_filter,
        "evaluation_rows": len(rows),
        "metrics": result["metrics"],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_csv(args.output_predictions, result["predictions"])

    try:
        temp_csv.unlink()
    except FileNotFoundError:
        pass

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
