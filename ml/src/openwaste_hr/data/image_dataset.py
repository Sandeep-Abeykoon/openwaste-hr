from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image

from openwaste_hr.data.manifest import load_manifest


class ManifestImageDataset:
    """
    Manifest-based image dataset for OpenWaste-HR.

    This class does not depend on PyTorch yet. It loads images using PIL and
    returns a dictionary containing the image and metadata.

    Later, this class can be wrapped or extended for PyTorch training.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path = ".",
        usage_filter: list[str] | None = None,
        label_column: str = "fine_label",
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.label_column = label_column
        self.transform = transform

        manifest = load_manifest(self.manifest_path)

        if usage_filter is not None:
            manifest = manifest[manifest["usage"].isin(usage_filter)].copy()

        if manifest.empty:
            raise ValueError("Dataset manifest is empty after applying filters.")

        if label_column not in manifest.columns:
            raise ValueError(f"Label column '{label_column}' not found in manifest.")

        self.manifest = manifest.reset_index(drop=True)

        label_names = sorted(self.manifest[label_column].astype(str).unique())
        self.label_to_id = {label: index for index, label in enumerate(label_names)}
        self.id_to_label = {index: label for label, index in self.label_to_id.items()}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.manifest):
            raise IndexError(f"Index out of range: {index}")

        row = self.manifest.iloc[index]
        image_path = self.project_root / str(row["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_name = str(row[self.label_column])
        label_id = self.label_to_id[label_name]

        return {
            "image": image,
            "label_id": label_id,
            "label_name": label_name,
            "sample_id": row["sample_id"],
            "image_path": str(row["image_path"]),
            "source_dataset": row["source_dataset"],
            "source_split": row["source_split"],
            "original_label": row["original_label"],
            "fine_label": row["fine_label"],
            "coarse_label": row["coarse_label"],
            "usage": row["usage"],
        }

    def get_label_distribution(self) -> pd.DataFrame:
        """
        Return count and percentage distribution for the selected label column.
        """
        counts = (
            self.manifest[self.label_column]
            .astype(str)
            .value_counts()
            .rename_axis(self.label_column)
            .reset_index(name="count")
        )

        counts["percentage"] = (counts["count"] / len(self.manifest) * 100).round(2)

        return counts

    def get_class_names(self) -> list[str]:
        """
        Return class names in internal ID order.
        """
        return [self.id_to_label[index] for index in sorted(self.id_to_label)]