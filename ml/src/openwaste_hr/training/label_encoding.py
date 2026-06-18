from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from openwaste_hr.utils.taxonomy import (
    get_coarse_labels,
    get_fine_labels,
    load_taxonomy,
)


def build_label_names_from_manifest(
    manifest: pd.DataFrame,
    taxonomy_path: str | Path,
    label_column: str = "fine_label",
) -> list[str]:
    """
    Build label names in taxonomy order, keeping only labels present in the manifest.

    For the first TrashNet baseline, this keeps the available labels only.
    Example:
    paper_cardboard, plastic, glass, metal, residual

    Organic and e_waste_battery are not included until actual training images
    for those labels are added.
    """
    if label_column not in manifest.columns:
        raise ValueError(f"Label column not found in manifest: {label_column}")

    taxonomy = load_taxonomy(taxonomy_path)

    if label_column == "fine_label":
        taxonomy_labels = get_fine_labels(taxonomy)
    elif label_column == "coarse_label":
        taxonomy_labels = get_coarse_labels(taxonomy)
    else:
        raise ValueError(
            "Only 'fine_label' and 'coarse_label' are supported for taxonomy-based encoding."
        )

    present_labels = set(manifest[label_column].astype(str).unique())
    known_taxonomy_labels = set(taxonomy_labels)

    unknown_labels = present_labels - known_taxonomy_labels
    if unknown_labels:
        raise ValueError(
            f"Manifest contains labels not present in taxonomy: {sorted(unknown_labels)}"
        )

    ordered_present_labels = [
        label for label in taxonomy_labels
        if label in present_labels
    ]

    if len(ordered_present_labels) < 2:
        raise ValueError("At least two labels are required for classification training.")

    return ordered_present_labels


def build_label_to_id(label_names: list[str]) -> dict[str, int]:
    """
    Convert label names into numeric class IDs.
    """
    if len(label_names) != len(set(label_names)):
        raise ValueError("Duplicate label names found.")

    return {label_name: index for index, label_name in enumerate(label_names)}


def build_id_to_label(label_to_id: dict[str, int]) -> dict[int, str]:
    """
    Convert numeric class IDs back to label names.
    """
    return {index: label_name for label_name, index in label_to_id.items()}


def encode_labels(
    labels: list[str],
    label_to_id: dict[str, int],
) -> list[int]:
    """
    Encode string labels into numeric IDs.
    """
    encoded: list[int] = []

    for label in labels:
        if label not in label_to_id:
            raise ValueError(f"Label not found in label_to_id mapping: {label}")

        encoded.append(label_to_id[label])

    return encoded


def save_label_mapping(
    label_names: list[str],
    output_path: str | Path,
) -> None:
    """
    Save class mapping as JSON so predictions can be decoded later.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_to_id = build_label_to_id(label_names)

    payload: dict[str, Any] = {
        "label_names": label_names,
        "label_to_id": label_to_id,
        "id_to_label": {str(index): label for label, index in label_to_id.items()},
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_label_mapping(mapping_path: str | Path) -> dict[str, Any]:
    """
    Load a saved class mapping JSON file.
    """
    mapping_path = Path(mapping_path)

    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")

    return json.loads(mapping_path.read_text(encoding="utf-8"))