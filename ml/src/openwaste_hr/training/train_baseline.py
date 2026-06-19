from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from openwaste_hr.data.manifest import load_manifest, validate_manifest
from openwaste_hr.models.baseline_cnn import create_baseline_cnn
from openwaste_hr.training.image_transforms import (
    build_eval_transform,
    build_train_transform,
)
from openwaste_hr.training.label_encoding import (
    build_label_names_from_manifest,
    build_label_to_id,
    save_label_mapping,
)
from openwaste_hr.training.torch_dataset import TorchManifestImageDataset
from openwaste_hr.utils.seed import set_global_seed


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
    Resolve training device.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def compute_class_weights(
    train_manifest: pd.DataFrame,
    label_column: str,
    label_to_id: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced training data.
    """
    counts = train_manifest[label_column].astype(str).value_counts().to_dict()

    total_samples = len(train_manifest)
    num_classes = len(label_to_id)

    weights = torch.ones(num_classes, dtype=torch.float32)

    for label_name, class_id in label_to_id.items():
        class_count = counts.get(label_name, 0)

        if class_count <= 0:
            raise ValueError(f"No training samples found for label: {label_name}")

        weights[class_id] = total_samples / (num_classes * class_count)

    return weights.to(device)


def build_dataloaders(
    project_root: Path,
    train_manifest_path: Path,
    val_manifest_path: Path,
    test_manifest_path: Path,
    label_to_id: dict[str, int],
    label_column: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test dataloaders.
    """
    train_dataset = TorchManifestImageDataset(
        manifest_path=train_manifest_path,
        project_root=project_root,
        label_to_id=label_to_id,
        usage_filter=["known_train"],
        label_column=label_column,
        transform=build_train_transform(image_size=image_size),
    )

    val_dataset = TorchManifestImageDataset(
        manifest_path=val_manifest_path,
        project_root=project_root,
        label_to_id=label_to_id,
        usage_filter=["known_val"],
        label_column=label_column,
        transform=build_eval_transform(image_size=image_size),
    )

    test_dataset = TorchManifestImageDataset(
        manifest_path=test_manifest_path,
        project_root=project_root,
        label_to_id=label_to_id,
        usage_filter=["known_test"],
        label_column=label_column,
        transform=build_eval_transform(image_size=image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Train for one epoch.
    """
    model.train()

    running_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)

        running_loss += loss.item() * images.size(0)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    total_seen = len(all_targets)

    if total_seen == 0:
        raise ValueError("No batches were processed during training.")

    return {
        "loss": float(running_loss / total_seen),
        "accuracy": float(accuracy_score(all_targets, all_predictions)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Evaluate model on validation or test data.
    """
    model.eval()

    running_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch["image"].to(device)
        targets = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        predictions = torch.argmax(logits, dim=1)

        running_loss += loss.item() * images.size(0)
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    total_seen = len(all_targets)

    if total_seen == 0:
        raise ValueError("No batches were processed during evaluation.")

    labels = list(range(num_classes))

    return {
        "loss": float(running_loss / total_seen),
        "accuracy": float(accuracy_score(all_targets, all_predictions)),
        "macro_f1": float(
            f1_score(
                all_targets,
                all_predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                all_targets,
                all_predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def save_checkpoint(
    model: nn.Module,
    label_names: list[str],
    config: dict[str, Any],
    output_path: str | Path,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """
    Save model checkpoint.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "label_names": label_names,
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }

    torch.save(payload, output_path)


def load_checkpoint_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load saved checkpoint weights into a model.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


def train_baseline(
    config_path: str | Path,
    project_root: str | Path,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    """
    Train the first closed-set baseline classifier.

    Best checkpoint selection is based on validation macro-F1 because the
    TrashNet baseline is class-imbalanced.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    output_config = config["outputs"]

    set_global_seed(int(training_config["seed"]), deterministic=False)

    train_manifest_path = resolve_path(project_root, paths["train_manifest"])
    val_manifest_path = resolve_path(project_root, paths["val_manifest"])
    test_manifest_path = resolve_path(project_root, paths["test_manifest"])
    taxonomy_path = resolve_path(project_root, paths["taxonomy"])
    checkpoint_dir = resolve_path(project_root, paths["output_checkpoint_dir"])
    metrics_dir = resolve_path(project_root, paths["output_metrics_dir"])

    train_manifest = load_manifest(train_manifest_path)
    val_manifest = load_manifest(val_manifest_path)
    test_manifest = load_manifest(test_manifest_path)

    validate_manifest(train_manifest, taxonomy_path)
    validate_manifest(val_manifest, taxonomy_path)
    validate_manifest(test_manifest, taxonomy_path)

    label_column = str(data_config["label_column"])

    label_names = build_label_names_from_manifest(
        manifest=train_manifest,
        taxonomy_path=taxonomy_path,
        label_column=label_column,
    )
    label_to_id = build_label_to_id(label_names)

    device = get_device(str(training_config["device"]))

    train_loader, val_loader, test_loader = build_dataloaders(
        project_root=project_root,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        test_manifest_path=test_manifest_path,
        label_to_id=label_to_id,
        label_column=label_column,
        image_size=int(data_config["image_size"]),
        batch_size=int(training_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
    )

    model = create_baseline_cnn(
        model_name=str(model_config["name"]),
        num_classes=len(label_names),
        pretrained=bool(model_config["pretrained"]),
        drop_rate=float(model_config["drop_rate"]),
    )
    model = model.to(device)

    class_weights = None
    if bool(training_config["use_class_weights"]):
        class_weights = compute_class_weights(
            train_manifest=train_manifest,
            label_column=label_column,
            label_to_id=label_to_id,
            device=device,
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    class_mapping_path = checkpoint_dir / output_config["class_mapping_json"]
    save_label_mapping(label_names, class_mapping_path)

    best_checkpoint_path = checkpoint_dir / output_config["best_checkpoint"]
    final_checkpoint_path = checkpoint_dir / output_config["final_checkpoint"]

    best_val_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    epochs = int(training_config["epochs"])
    early_stopping_patience = int(training_config.get("early_stopping_patience", 7))
    early_stopping_min_delta = float(training_config.get("early_stopping_min_delta", 0.001))
    monitor_metric = str(training_config.get("monitor_metric", "val_macro_f1"))

    if monitor_metric != "val_macro_f1":
        raise ValueError(
            "Only monitor_metric='val_macro_f1' is currently supported for baseline training."
        )

    print(f"Device: {device}")
    print(f"Classes: {label_names}")
    print(f"Epochs: {epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Early stopping min delta: {early_stopping_min_delta}")
    print(f"Monitoring: {monitor_metric}")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=len(label_names),
            max_batches=max_eval_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "best_epoch_so_far": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_macro_f1={row['val_macro_f1']:.4f}"
        )

        improvement = val_metrics["macro_f1"] - best_val_macro_f1

        if improvement > early_stopping_min_delta:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                label_names=label_names,
                config=config,
                output_path=best_checkpoint_path,
                epoch=epoch,
                metrics=val_metrics,
            )

            print(
                f"New best checkpoint saved at epoch {epoch} "
                f"with val_macro_f1={best_val_macro_f1:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No significant improvement. "
                f"Patience counter: {epochs_without_improvement}/{early_stopping_patience}"
            )

        history[-1]["best_epoch_so_far"] = best_epoch
        history[-1]["epochs_without_improvement"] = epochs_without_improvement

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch: {best_epoch}, "
                f"best val_macro_f1={best_val_macro_f1:.4f}"
            )
            break

    final_epoch = int(history[-1]["epoch"])

    save_checkpoint(
        model=model,
        label_names=label_names,
        config=config,
        output_path=final_checkpoint_path,
        epoch=final_epoch,
        metrics=history[-1],
    )

    metrics_csv_path = metrics_dir / output_config["metrics_csv"]
    pd.DataFrame(history).to_csv(metrics_csv_path, index=False)

    # Test evaluation must use the best validation checkpoint, not the last epoch model.
    best_checkpoint = load_checkpoint_weights(
        model=model,
        checkpoint_path=best_checkpoint_path,
        device=device,
    )

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=len(label_names),
        max_batches=max_eval_batches,
    )

    test_metrics_payload = {
        "label_names": label_names,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_metrics": best_checkpoint["metrics"],
        "test_metrics": test_metrics,
    }

    test_metrics_path = metrics_dir / output_config["test_metrics_json"]
    test_metrics_path.write_text(
        json.dumps(test_metrics_payload, indent=2),
        encoding="utf-8",
    )

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_val_macro_f1:.4f}")
    print(f"Test accuracy using best checkpoint: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1 using best checkpoint: {test_metrics['macro_f1']:.4f}")
    print(f"Training metrics CSV: {metrics_csv_path}")
    print(f"Test metrics JSON: {test_metrics_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")

    return {
        "label_names": label_names,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics,
        "metrics_csv": str(metrics_csv_path),
        "test_metrics_json": str(test_metrics_path),
        "checkpoint_dir": str(checkpoint_dir),
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the first OpenWaste-HR closed-set baseline model."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/train_baseline_trashnet.yaml",
        help="Path to baseline training YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )

    args = parser.parse_args()

    train_baseline(
        config_path=args.config,
        project_root=args.project_root,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )


if __name__ == "__main__":
    main()