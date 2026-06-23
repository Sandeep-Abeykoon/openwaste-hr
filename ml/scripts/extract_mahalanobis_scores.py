from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from train_image_classifier import WasteImageDataset, build_model, build_transforms


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

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = list(checkpoint.get("class_names", config["labels"]["known_classes"]))

    model = build_model(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def find_last_linear_layer(model: nn.Module) -> tuple[str, nn.Linear]:
    last_name = ""
    last_layer: nn.Linear | None = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_layer = module

    if last_layer is None:
        raise RuntimeError("Could not find a Linear layer for embedding extraction.")

    return last_name, last_layer


def get_true_label(row: dict[str, Any]) -> str:
    for key in ["true_label", "canonical_label", "label", "reviewed_fine_label"]:
        value = str(row.get(key, "")).strip()
        if value:
            return value

    return ""


def extract_embeddings_and_predictions(
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
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = read_csv(csv_path)

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

    layer_name, last_linear = find_last_linear_layer(model)

    captured_embeddings: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured_embeddings.append(inputs[0].detach().cpu())

    handle = last_linear.register_forward_pre_hook(hook)

    print(
        f"Extracting embeddings and predictions for {split_name} | "
        f"rows={len(rows)} | hook_layer={layer_name}",
        flush=True,
    )

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                images = batch["image"].to(device)
                logits = model(images)

                all_logits.append(logits.detach().cpu())

                if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
                    print(
                        f"{split_name}: batch {batch_index}/{len(loader)}",
                        flush=True,
                    )

    finally:
        handle.remove()

    embeddings = torch.cat(captured_embeddings, dim=0).numpy()
    logits_tensor = torch.cat(all_logits, dim=0)

    if embeddings.shape[0] != len(rows):
        raise RuntimeError(
            f"Embedding row count mismatch for {split_name}: "
            f"embeddings={embeddings.shape[0]}, rows={len(rows)}"
        )

    if logits_tensor.shape[0] != len(rows):
        raise RuntimeError(
            f"Logit row count mismatch for {split_name}: "
            f"logits={logits_tensor.shape[0]}, rows={len(rows)}"
        )

    probabilities = F.softmax(logits_tensor, dim=1)
    confidence_tensor, pred_index_tensor = torch.max(probabilities, dim=1)
    max_logit_tensor, _ = torch.max(logits_tensor, dim=1)
    energy_tensor = -torch.logsumexp(logits_tensor, dim=1)

    output_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        true_label = get_true_label(row)
        pred_index = int(pred_index_tensor[index].item())
        pred_label = class_names[pred_index]

        updated: dict[str, Any] = dict(row)
        updated["split"] = split_name
        updated["true_label"] = true_label
        updated["pred_label"] = pred_label
        updated["pred_index"] = pred_index
        updated["confidence"] = float(confidence_tensor[index].item())
        updated["max_logit"] = float(max_logit_tensor[index].item())
        updated["energy"] = float(energy_tensor[index].item())

        for class_index, class_name in enumerate(class_names):
            updated[f"prob_{class_name}"] = float(probabilities[index, class_index].item())

        output_rows.append(updated)

    return embeddings, output_rows


def fit_mahalanobis_model(
    *,
    train_embeddings: np.ndarray,
    train_rows: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_embeddings)

    class_means: dict[str, np.ndarray] = {}

    for class_name in class_names:
        class_indices = [
            index for index, row in enumerate(train_rows)
            if get_true_label(row) == class_name
        ]

        if not class_indices:
            raise ValueError(f"No training rows found for class: {class_name}")

        class_means[class_name] = np.mean(train_z[class_indices], axis=0)

    covariance_model = LedoitWolf()
    covariance_model.fit(train_z)

    precision = covariance_model.precision_

    return {
        "scaler": scaler,
        "class_means": class_means,
        "precision": precision,
        "shrinkage": float(covariance_model.shrinkage_),
    }


def compute_mahalanobis_scores(
    *,
    embeddings: np.ndarray,
    class_names: list[str],
    mahalanobis_model: dict[str, Any],
    batch_size: int = 1024,
) -> tuple[np.ndarray, list[str]]:
    scaler: StandardScaler = mahalanobis_model["scaler"]
    class_means: dict[str, np.ndarray] = mahalanobis_model["class_means"]
    precision: np.ndarray = mahalanobis_model["precision"]

    z = scaler.transform(embeddings)

    means = np.stack(
        [class_means[class_name] for class_name in class_names],
        axis=0,
    )

    min_distances: list[np.ndarray] = []
    nearest_classes: list[str] = []

    for start in range(0, z.shape[0], batch_size):
        end = min(start + batch_size, z.shape[0])
        batch = z[start:end]

        diff = batch[:, None, :] - means[None, :, :]

        distances = np.einsum("ncd,dk,nck->nc", diff, precision, diff)
        distances = np.maximum(distances, 0.0)

        nearest_indices = np.argmin(distances, axis=1)
        batch_min = distances[np.arange(distances.shape[0]), nearest_indices]

        min_distances.append(batch_min)
        nearest_classes.extend([class_names[index] for index in nearest_indices])

    return np.concatenate(min_distances, axis=0), nearest_classes


def is_correct(row: dict[str, Any]) -> bool:
    return str(row.get("true_label", "")).strip() == str(row.get("pred_label", "")).strip()


def evaluate_threshold(
    *,
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    known_accept = known_scores >= threshold
    unknown_accept = unknown_scores >= threshold

    accepted_known_rows = [
        row for row, accepted in zip(known_rows, known_accept)
        if bool(accepted)
    ]

    known_coverage = float(np.mean(known_accept)) if len(known_accept) else 0.0
    known_rejection_rate = 1.0 - known_coverage

    unknown_acceptance_rate = float(np.mean(unknown_accept)) if len(unknown_accept) else 0.0
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
        "known_rows": len(known_rows),
        "unknown_rows": len(unknown_rows),
        "known_coverage": float(known_coverage),
        "known_rejection_rate": float(known_rejection_rate),
        "unknown_rejection_rate": float(unknown_rejection_rate),
        "unknown_acceptance_rate": float(unknown_acceptance_rate),
        "false_acceptance_rate": float(unknown_acceptance_rate),
        "false_rejection_rate": float(known_rejection_rate),
        "known_accuracy_all": float(known_accuracy_all),
        "known_accuracy_accepted": float(known_accuracy_accepted),
        "selective_risk": float(selective_risk),
        "binary_balanced_accuracy": float(binary_balanced_accuracy),
    }


def select_threshold(
    *,
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_known_coverage_floor: float,
) -> dict[str, Any]:
    all_scores = np.concatenate([known_scores, unknown_scores])
    thresholds = np.unique(all_scores)

    if len(thresholds) > 1000:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, 1000))

    candidates = [
        evaluate_threshold(
            known_rows=known_rows,
            unknown_rows=unknown_rows,
            known_scores=known_scores,
            unknown_scores=unknown_scores,
            threshold=float(threshold),
        )
        for threshold in thresholds
    ]

    feasible = [
        result for result in candidates
        if result["known_coverage"] >= target_known_coverage_floor
    ]

    if feasible:
        selected = max(
            feasible,
            key=lambda result: (
                result["unknown_rejection_rate"],
                result["known_accuracy_accepted"],
                result["binary_balanced_accuracy"],
                result["known_coverage"],
            ),
        )
        selected["threshold_selection_mode"] = (
            "minimize_false_acceptance_subject_to_known_coverage_floor"
        )
        return selected

    selected = max(
        candidates,
        key=lambda result: (
            result["binary_balanced_accuracy"],
            result["known_accuracy_accepted"],
            result["unknown_rejection_rate"],
            result["known_coverage"],
        ),
    )
    selected["threshold_selection_mode"] = "fallback_best_binary_balanced_accuracy"
    return selected


def add_scores_to_rows(
    *,
    rows: list[dict[str, Any]],
    min_distances: np.ndarray,
    nearest_classes: list[str],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []

    for row, distance, nearest_class in zip(rows, min_distances, nearest_classes):
        updated: dict[str, Any] = dict(row)
        updated["mahalanobis_min_distance"] = float(distance)
        updated["mahalanobis_knownness"] = float(-distance)
        updated["mahalanobis_nearest_class"] = nearest_class
        output_rows.append(updated)

    return output_rows


def split_rows(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("split", "") == split_name
    ]


def score_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [float(row["mahalanobis_knownness"]) for row in rows],
        dtype=float,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Mahalanobis feature-distance scores for OpenWaste-HR."
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
        "--output-dir",
        type=Path,
        default=Path("ml/outputs/fusion_gate/stage_04_mahalanobis_v1"),
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-known-coverage-floor", type=float, default=0.75)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = read_yaml(args.config)
    data_config = config["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}", flush=True)

    model, class_names = load_model(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    label_to_index = {label: index for index, label in enumerate(class_names)}

    transform = build_transforms(
        Path(config["preprocessing"]["config_path"]),
        train=False,
    )

    train_embeddings, train_rows = extract_embeddings_and_predictions(
        split_name="known_train",
        csv_path=Path(data_config["train_csv"]),
        model=model,
        class_names=class_names,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    known_val_embeddings, known_val_rows = extract_embeddings_and_predictions(
        split_name="known_val",
        csv_path=Path(data_config["val_csv"]),
        model=model,
        class_names=class_names,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    known_test_embeddings, known_test_rows = extract_embeddings_and_predictions(
        split_name="known_test",
        csv_path=Path(data_config["test_csv"]),
        model=model,
        class_names=class_names,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unknown_val_embeddings, unknown_val_rows = extract_embeddings_and_predictions(
        split_name="unknown_val",
        csv_path=Path(data_config["unknown_val_csv"]),
        model=model,
        class_names=class_names,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    unknown_test_embeddings, unknown_test_rows = extract_embeddings_and_predictions(
        split_name="unknown_test",
        csv_path=Path(data_config["unknown_test_csv"]),
        model=model,
        class_names=class_names,
        label_to_index=label_to_index,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Fitting Mahalanobis model on known_train embeddings...", flush=True)

    mahalanobis_model = fit_mahalanobis_model(
        train_embeddings=train_embeddings,
        train_rows=train_rows,
        class_names=class_names,
    )

    all_outputs: list[dict[str, Any]] = []

    split_data = [
        ("known_val", known_val_embeddings, known_val_rows),
        ("known_test", known_test_embeddings, known_test_rows),
        ("unknown_val", unknown_val_embeddings, unknown_val_rows),
        ("unknown_test", unknown_test_embeddings, unknown_test_rows),
    ]

    for split_name, embeddings, rows in split_data:
        print(f"Computing Mahalanobis scores for {split_name}...", flush=True)

        min_distances, nearest_classes = compute_mahalanobis_scores(
            embeddings=embeddings,
            class_names=class_names,
            mahalanobis_model=mahalanobis_model,
        )

        all_outputs.extend(
            add_scores_to_rows(
                rows=rows,
                min_distances=min_distances,
                nearest_classes=nearest_classes,
            )
        )

    known_val_scored = split_rows(all_outputs, "known_val")
    unknown_val_scored = split_rows(all_outputs, "unknown_val")
    known_test_scored = split_rows(all_outputs, "known_test")
    unknown_test_scored = split_rows(all_outputs, "unknown_test")

    known_val_scores = score_array(known_val_scored)
    unknown_val_scores = score_array(unknown_val_scored)
    known_test_scores = score_array(known_test_scored)
    unknown_test_scores = score_array(unknown_test_scored)

    selected_threshold = select_threshold(
        known_rows=known_val_scored,
        unknown_rows=unknown_val_scored,
        known_scores=known_val_scores,
        unknown_scores=unknown_val_scores,
        target_known_coverage_floor=args.target_known_coverage_floor,
    )

    threshold = float(selected_threshold["threshold"])

    validation_result = evaluate_threshold(
        known_rows=known_val_scored,
        unknown_rows=unknown_val_scored,
        known_scores=known_val_scores,
        unknown_scores=unknown_val_scores,
        threshold=threshold,
    )

    test_result = evaluate_threshold(
        known_rows=known_test_scored,
        unknown_rows=unknown_test_scored,
        known_scores=known_test_scores,
        unknown_scores=unknown_test_scores,
        threshold=threshold,
    )

    validation_auroc = float(
        roc_auc_score(
            [1] * len(known_val_scores) + [0] * len(unknown_val_scores),
            np.concatenate([known_val_scores, unknown_val_scores]),
        )
    )

    test_auroc = float(
        roc_auc_score(
            [1] * len(known_test_scores) + [0] * len(unknown_test_scores),
            np.concatenate([known_test_scores, unknown_test_scores]),
        )
    )

    validation_result["auroc_known_vs_unknown"] = validation_auroc
    test_result["auroc_known_vs_unknown"] = test_auroc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = args.output_dir / "mahalanobis_scores_v1.csv"
    summary_path = args.output_dir / "mahalanobis_summary_v1.json"
    model_path = args.output_dir / "mahalanobis_model_v1.joblib"

    write_csv(scores_path, all_outputs)
    joblib.dump(mahalanobis_model, model_path)

    summary = {
        "experiment": "stage_04_mahalanobis_v1",
        "base_model": config["experiment"]["name"],
        "checkpoint": str(args.checkpoint),
        "embedding_layer": find_last_linear_layer(model)[0],
        "embedding_dimension": int(train_embeddings.shape[1]),
        "class_names": class_names,
        "fit_split": "known_train",
        "fit_rows": len(train_rows),
        "target_known_coverage_floor": args.target_known_coverage_floor,
        "covariance_shrinkage": mahalanobis_model["shrinkage"],
        "selected_threshold_from_validation": selected_threshold,
        "validation_result": validation_result,
        "test_result": test_result,
        "output_files": {
            "scores": str(scores_path),
            "summary": str(summary_path),
            "model": str(model_path),
        },
        "rule": (
            "Mahalanobis class statistics are fitted using known_train only. "
            "Threshold is selected using known_val + unknown_val only. "
            "Final reporting uses known_test + unknown_test."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
