from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


class WasteImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        label_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.rows = read_csv_rows(csv_path)
        self.label_to_index = label_to_index
        self.transform = transform

        if not self.rows:
            raise ValueError(f"Dataset CSV is empty: {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(row["image_path"])

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        label_name = row["canonical_label"]
        label_index = self.label_to_index.get(label_name, -1)

        return {
            "image": image_tensor,
            "label_index": torch.tensor(label_index, dtype=torch.long),
            "label_name": label_name,
            "image_path": str(image_path),
            "source_dataset": row.get("source_dataset", ""),
            "source_label": row.get("source_label", ""),
        }


def build_transforms(preprocessing_config_path: Path, train: bool) -> transforms.Compose:
    config = read_yaml(preprocessing_config_path)

    preprocessing = config["preprocessing"]
    normalization = preprocessing["normalization"]

    mean = normalization["mean"]
    std = normalization["std"]

    if train:
        aug = config["training_augmentation"]

        random_crop = aug["random_resized_crop"]
        rotation = aug["rotation"]
        color_jitter = aug["color_jitter"]
        affine = aug["affine"]
        horizontal_flip = aug["horizontal_flip"]

        transform_steps: list[Any] = []

        if random_crop["enabled"]:
            transform_steps.append(
                transforms.RandomResizedCrop(
                    size=random_crop["size"],
                    scale=(random_crop["scale_min"], random_crop["scale_max"]),
                    ratio=(random_crop["ratio_min"], random_crop["ratio_max"]),
                )
            )
        else:
            transform_steps.append(transforms.Resize((preprocessing["image_size"], preprocessing["image_size"])))

        if horizontal_flip["enabled"]:
            transform_steps.append(transforms.RandomHorizontalFlip(p=horizontal_flip["probability"]))

        if rotation["enabled"]:
            transform_steps.append(transforms.RandomRotation(degrees=rotation["degrees"]))

        if affine["enabled"]:
            transform_steps.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=tuple(affine["translate"]),
                    scale=tuple(affine["scale"]),
                )
            )

        if color_jitter["enabled"]:
            transform_steps.append(
                transforms.ColorJitter(
                    brightness=color_jitter["brightness"],
                    contrast=color_jitter["contrast"],
                    saturation=color_jitter["saturation"],
                    hue=color_jitter["hue"],
                )
            )

        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        return transforms.Compose(transform_steps)

    eval_cfg = config["evaluation_preprocessing"]

    return transforms.Compose(
        [
            transforms.Resize(eval_cfg["resize_size"]),
            transforms.CenterCrop(eval_cfg["center_crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    model_config = config["model"]
    backbone = model_config["backbone"]
    pretrained = bool(model_config.get("pretrained", True))
    dropout = float(model_config.get("dropout", 0.2))

    if backbone != "mobilenet_v3_large":
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Current script supports mobilenet_v3_large."
        )

    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    training_config = config["training"]

    optimizer_name = training_config["optimizer"].lower()
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    training_config = config["training"]
    scheduler_name = str(training_config.get("scheduler", "")).lower()

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training_config["epochs"]),
        )

    if scheduler_name in {"", "none", "null"}:
        return None

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label_index"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    class_names: list[str],
    calculate_known_metrics: bool,
) -> dict[str, Any]:
    model.eval()

    prediction_rows: list[dict[str, Any]] = []
    valid_targets: list[int] = []
    valid_predictions: list[int] = []

    total_loss = 0.0
    total_loss_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label_index"].to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

        confidence, predictions = torch.max(probabilities, dim=1)
        max_logit, _ = torch.max(logits, dim=1)
        energy = -torch.logsumexp(logits, dim=1)

        valid_mask = labels >= 0

        if criterion is not None and bool(valid_mask.all()):
            loss = criterion(logits, labels)
            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_loss_samples += batch_size

        if calculate_known_metrics:
            valid_targets.extend(labels[valid_mask].cpu().numpy().tolist())
            valid_predictions.extend(predictions[valid_mask].cpu().numpy().tolist())

        for i in range(images.size(0)):
            row: dict[str, Any] = {
                "image_path": batch["image_path"][i],
                "source_dataset": batch["source_dataset"][i],
                "source_label": batch["source_label"][i],
                "true_label": batch["label_name"][i],
                "true_index": int(labels[i].cpu().item()),
                "pred_label": class_names[int(predictions[i].cpu().item())],
                "pred_index": int(predictions[i].cpu().item()),
                "confidence": float(confidence[i].cpu().item()),
                "max_logit": float(max_logit[i].cpu().item()),
                "energy": float(energy[i].cpu().item()),
            }

            for class_index, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(probabilities[i, class_index].cpu().item())

            if int(labels[i].cpu().item()) >= 0:
                row["is_correct"] = str(int(labels[i].cpu().item()) == int(predictions[i].cpu().item())).lower()
            else:
                row["is_correct"] = ""

            prediction_rows.append(row)

    metrics: dict[str, Any] = {}

    if calculate_known_metrics:
        metrics["accuracy"] = float(accuracy_score(valid_targets, valid_predictions))
        metrics["macro_f1"] = float(f1_score(valid_targets, valid_predictions, average="macro"))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(valid_targets, valid_predictions))
        metrics["confusion_matrix"] = confusion_matrix(
            valid_targets,
            valid_predictions,
            labels=list(range(len(class_names))),
        ).tolist()

    if total_loss_samples > 0:
        metrics["loss"] = total_loss / total_loss_samples

    return {
        "metrics": metrics,
        "predictions": prediction_rows,
    }


def save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(
    path: Path,
    matrix: list[list[int]],
    class_names: list[str],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    matrix_np = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix_np)
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            ax.text(j, i, str(matrix_np[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict[str, Any],
    class_names: list[str],
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "config": config,
        "best_metric": best_metric,
    }

    torch.save(checkpoint, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpenWaste-HR image classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/configs/train_stage_01_trashnet.yaml"),
        help="Training configuration YAML path.",
    )
    args = parser.parse_args()

    config = read_yaml(args.config)

    seed = int(config["training"]["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = list(config["labels"]["known_classes"])
    label_to_index = {label: index for index, label in enumerate(class_names)}

    preprocessing_config_path = Path(config["preprocessing"]["config_path"])

    train_transform = build_transforms(preprocessing_config_path, train=True)
    eval_transform = build_transforms(preprocessing_config_path, train=False)

    train_dataset = WasteImageDataset(
        csv_path=Path(config["data"]["train_csv"]),
        label_to_index=label_to_index,
        transform=train_transform,
    )
    val_dataset = WasteImageDataset(
        csv_path=Path(config["data"]["val_csv"]),
        label_to_index=label_to_index,
        transform=eval_transform,
    )
    test_dataset = WasteImageDataset(
        csv_path=Path(config["data"]["test_csv"]),
        label_to_index=label_to_index,
        transform=eval_transform,
    )
    unknown_val_dataset = WasteImageDataset(
        csv_path=Path(config["data"]["unknown_val_csv"]),
        label_to_index=label_to_index,
        transform=eval_transform,
    )
    unknown_test_dataset = WasteImageDataset(
        csv_path=Path(config["data"]["unknown_test_csv"]),
        label_to_index=label_to_index,
        transform=eval_transform,
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    unknown_val_loader = DataLoader(
        unknown_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    unknown_test_loader = DataLoader(
        unknown_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config, num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    output_dir = Path(config["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(config["outputs"]["model_path"])
    metrics_path = Path(config["outputs"]["metrics_path"])
    predictions_path = Path(config["outputs"]["predictions_path"])
    confusion_matrix_path = Path(config["outputs"]["confusion_matrix_path"])

    epochs = int(config["training"]["epochs"])
    early_stopping = config["training"].get("early_stopping", {})
    patience = int(early_stopping.get("patience", epochs))

    best_val_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_result = predict_dataset(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            calculate_known_metrics=True,
        )
        val_metrics = val_result["metrics"]

        if scheduler is not None:
            scheduler.step()

        current_val_macro_f1 = float(val_metrics["macro_f1"])

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        improved = current_val_macro_f1 > best_val_macro_f1

        if improved:
            previous_best = best_val_macro_f1
            best_val_macro_f1 = current_val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=model_path,
                model=model,
                config=config,
                class_names=class_names,
                best_metric=best_val_macro_f1,
            )

            if previous_best < 0:
                early_stopping_message = (
                    f"IMPROVED | first best model saved | "
                    f"best_epoch={best_epoch} | "
                    f"best_val_macro_f1={best_val_macro_f1:.4f} | "
                    f"patience=0/{patience}"
                )
            else:
                improvement_amount = best_val_macro_f1 - previous_best
                early_stopping_message = (
                    f"IMPROVED | +{improvement_amount:.4f} macro-F1 | "
                    f"best model saved | "
                    f"best_epoch={best_epoch} | "
                    f"best_val_macro_f1={best_val_macro_f1:.4f} | "
                    f"patience=0/{patience}"
                )
        else:
            epochs_without_improvement += 1
            early_stopping_message = (
                f"NO IMPROVEMENT | "
                f"best_epoch={best_epoch} | "
                f"best_val_macro_f1={best_val_macro_f1:.4f} | "
                f"patience={epochs_without_improvement}/{patience}"
            )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"{early_stopping_message}"
        )

        if early_stopping.get("enabled", True) and epochs_without_improvement >= patience:
            print(
                f"EARLY STOPPING TRIGGERED | "
                f"epoch={epoch} | "
                f"no improvement for {epochs_without_improvement} epoch(s) | "
                f"best_epoch={best_epoch} | "
                f"best_val_macro_f1={best_val_macro_f1:.4f}"
            )
            break

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_result = predict_dataset(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        calculate_known_metrics=True,
    )
    test_result = predict_dataset(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        calculate_known_metrics=True,
    )
    unknown_val_result = predict_dataset(
        model=model,
        loader=unknown_val_loader,
        criterion=None,
        device=device,
        class_names=class_names,
        calculate_known_metrics=False,
    )
    unknown_test_result = predict_dataset(
        model=model,
        loader=unknown_test_loader,
        criterion=None,
        device=device,
        class_names=class_names,
        calculate_known_metrics=False,
    )

    all_predictions = []
    for split_name, result in [
        ("known_val", val_result),
        ("known_test", test_result),
        ("unknown_val", unknown_val_result),
        ("unknown_test", unknown_test_result),
    ]:
        for row in result["predictions"]:
            updated = dict(row)
            updated["split"] = split_name
            all_predictions.append(updated)

    save_predictions(predictions_path, all_predictions)

    save_confusion_matrix(
        path=confusion_matrix_path,
        matrix=test_result["metrics"]["confusion_matrix"],
        class_names=class_names,
        title="Stage 1 TrashNet Baseline Confusion Matrix",
    )

    metrics_output = {
        "experiment": config["experiment"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "class_names": class_names,
        "history": history,
        "final_val_metrics": val_result["metrics"],
        "final_test_metrics": test_result["metrics"],
        "unknown_val_prediction_count": len(unknown_val_result["predictions"]),
        "unknown_test_prediction_count": len(unknown_test_result["predictions"]),
        "outputs": config["outputs"],
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_val_macro_f1:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()


