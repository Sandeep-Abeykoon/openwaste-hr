from pathlib import Path
from typing import Any

import pandas as pd

from openwaste_hr.utils.taxonomy import (
    get_coarse_labels,
    get_fine_labels,
    load_taxonomy,
)


REQUIRED_MANIFEST_COLUMNS = [
    "sample_id",
    "source_dataset",
    "source_split",
    "image_path",
    "original_label",
    "fine_label",
    "coarse_label",
    "is_known",
    "usage",
    "license_notes",
]

ALLOWED_USAGE_VALUES = {
    "known_train",
    "known_val",
    "known_test",
    "unknown_test",
    "local_unknown",
    "active_learning_candidate",
}


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """
    Load a dataset manifest CSV file.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    return pd.read_csv(manifest_path)


def _normalise_bool(value: Any) -> bool:
    """
    Convert CSV boolean values into Python booleans.
    """
    if isinstance(value, bool):
        return value

    value_text = str(value).strip().lower()

    if value_text == "true":
        return True

    if value_text == "false":
        return False

    raise ValueError(f"Invalid boolean value for is_known: {value}")


def validate_manifest_columns(manifest: pd.DataFrame) -> bool:
    """
    Check that the manifest contains all required columns.
    """
    missing_columns = [
        column for column in REQUIRED_MANIFEST_COLUMNS
        if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")

    return True


def validate_manifest(
    manifest: pd.DataFrame,
    taxonomy_path: str | Path,
) -> bool:
    """
    Validate the dataset manifest against the OpenWaste-HR taxonomy.

    This does not check whether image files physically exist yet.
    That will be done later after datasets are downloaded.
    """
    validate_manifest_columns(manifest)

    taxonomy = load_taxonomy(taxonomy_path)
    known_fine_labels = set(get_fine_labels(taxonomy))
    allowed_coarse_labels = set(get_coarse_labels(taxonomy))

    invalid_usage = set(manifest["usage"]) - ALLOWED_USAGE_VALUES
    if invalid_usage:
        raise ValueError(f"Invalid usage values found: {sorted(invalid_usage)}")

    for row_index, row in manifest.iterrows():
        is_known = _normalise_bool(row["is_known"])
        fine_label = str(row["fine_label"]).strip()
        coarse_label = str(row["coarse_label"]).strip()
        usage = str(row["usage"]).strip()

        if is_known:
            if fine_label not in known_fine_labels:
                raise ValueError(
                    f"Row {row_index}: known sample has invalid fine label '{fine_label}'."
                )

            if coarse_label not in allowed_coarse_labels:
                raise ValueError(
                    f"Row {row_index}: known sample has invalid coarse label '{coarse_label}'."
                )

            if usage not in {"known_train", "known_val", "known_test"}:
                raise ValueError(
                    f"Row {row_index}: known sample has invalid usage '{usage}'."
                )

        else:
            if usage not in {
                "unknown_test",
                "local_unknown",
                "active_learning_candidate",
            }:
                raise ValueError(
                    f"Row {row_index}: unknown sample has invalid usage '{usage}'."
                )

            if fine_label not in {"unknown", "manual_review"}:
                raise ValueError(
                    f"Row {row_index}: unknown sample should use fine_label "
                    f"'unknown' or 'manual_review', got '{fine_label}'."
                )

            if coarse_label not in allowed_coarse_labels:
                raise ValueError(
                    f"Row {row_index}: unknown sample has invalid coarse label '{coarse_label}'."
                )

    return True
