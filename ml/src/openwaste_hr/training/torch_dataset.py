from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from openwaste_hr.data.manifest import load_manifest


class TorchManifestImageDataset(Dataset):
    """
    PyTorch dataset built from an OpenWaste-HR manifest CSV.

    Each sample returns:
    - image tensor
    - numeric label
    - useful metadata
    """

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        label_to_id: dict[str, int],
        usage_filter: list[str] | None = None,
        label_column: str = "fine_label",
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.label_to_id = label_to_id
        self.label_column = label_column
        self.transform = transform

        manifest = load_manifest(self.manifest_path)

        if usage_filter is not None:
            manifest = manifest[manifest["usage"].isin(usage_filter)].copy()

        if manifest.empty:
            raise ValueError("Dataset manifest is empty after applying filters.")

        if label_column not in manifest.columns:
            raise ValueError(f"Label column '{label_column}' not found in manifest.")

        unknown_labels = set(manifest[label_column].astype(str).unique()) - set(label_to_id)
        if unknown_labels:
            raise ValueError(
                f"Manifest contains labels missing from label_to_id: {sorted(unknown_labels)}"
            )

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

        label_name = str(row[self.label_column])
        label_id = self.label_to_id[label_name]

        return {
            "image": image,
            "label": torch.tensor(label_id, dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "image_path": str(row["image_path"]),
            "label_name": label_name,
            "source_dataset": str(row["source_dataset"]),
            "source_split": str(row["source_split"]),
            "original_label": str(row["original_label"]),
            "fine_label": str(row["fine_label"]),
            "coarse_label": str(row["coarse_label"]),
            "usage": str(row["usage"]),
        }