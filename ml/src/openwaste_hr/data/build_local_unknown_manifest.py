from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from openwaste_hr.data.manifest import validate_manifest


REQUIRED_COLLECTION_COLUMNS = [
    "sample_id",
    "image_path",
    "capture_date",
    "location_context",
    "object_description",
    "why_unknown_or_difficult",
    "lighting_condition",
    "background_condition",
    "human_note",
    "usage",
]


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


def load_collection_sheet(collection_csv: str | Path) -> pd.DataFrame:
    """
    Load the local unknown collection sheet.
    """
    collection_csv = Path(collection_csv)

    if not collection_csv.exists():
        raise FileNotFoundError(
            f"Local unknown collection sheet not found: {collection_csv}"
        )

    return pd.read_csv(collection_csv)


def validate_collection_sheet(
    collection_df: pd.DataFrame,
    project_root: str | Path,
    allowed_usage: list[str],
    require_image_files_exist: bool,
) -> bool:
    """
    Validate local unknown collection metadata.
    """
    missing_columns = [
        column for column in REQUIRED_COLLECTION_COLUMNS
        if column not in collection_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Collection sheet is missing required columns: {missing_columns}"
        )

    if collection_df.empty:
        raise ValueError("Collection sheet is empty.")

    duplicate_sample_ids = collection_df[
        collection_df["sample_id"].astype(str).duplicated()
    ]

    if not duplicate_sample_ids.empty:
        raise ValueError("Duplicate sample_id values found in collection sheet.")

    allowed_usage_set = set(allowed_usage)
    usage_values = set(collection_df["usage"].astype(str).str.strip())
    invalid_usage = usage_values - allowed_usage_set

    if invalid_usage:
        raise ValueError(f"Invalid usage values found: {sorted(invalid_usage)}")

    empty_paths = collection_df[
        collection_df["image_path"].astype(str).str.strip() == ""
    ]

    if not empty_paths.empty:
        raise ValueError("Collection sheet contains empty image_path values.")

    if require_image_files_exist:
        project_root = Path(project_root)

        missing_paths = []
        for _, row in collection_df.iterrows():
            image_path = project_root / str(row["image_path"])
            if not image_path.exists():
                missing_paths.append(str(row["image_path"]))

        if missing_paths:
            preview = missing_paths[:10]
            raise FileNotFoundError(
                "Some local unknown image files do not exist. "
                f"First missing paths: {preview}"
            )

    return True


def build_local_unknown_manifest_dataframe(
    collection_df: pd.DataFrame,
    rules: dict[str, Any],
) -> pd.DataFrame:
    """
    Convert local unknown collection metadata into OpenWaste-HR manifest format.
    """
    rows: list[dict[str, Any]] = []

    for _, row in collection_df.iterrows():
        usage = str(row["usage"]).strip()

        rows.append(
            {
                "sample_id": str(row["sample_id"]).strip(),
                "source_dataset": str(rules["source_dataset"]),
                "source_split": usage,
                "image_path": str(row["image_path"]).strip(),
                "original_label": str(rules["original_label"]),
                "fine_label": str(rules["fine_label"]),
                "coarse_label": str(rules["coarse_label"]),
                "is_known": str(bool(rules["is_known"])).lower(),
                "usage": usage,
                "license_notes": str(rules["license_notes"]),
                "capture_date": str(row["capture_date"]),
                "location_context": str(row["location_context"]),
                "object_description": str(row["object_description"]),
                "why_unknown_or_difficult": str(row["why_unknown_or_difficult"]),
                "lighting_condition": str(row["lighting_condition"]),
                "background_condition": str(row["background_condition"]),
                "human_note": str(row["human_note"]),
            }
        )

    return pd.DataFrame(rows)


def build_local_unknown_manifest(
    config_path: str | Path,
    project_root: str | Path,
) -> pd.DataFrame:
    """
    Build local unknown manifest from collection sheet.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    rules = config["rules"]

    collection_csv = project_root / paths["collection_csv"]
    output_manifest = project_root / paths["output_manifest"]
    taxonomy_path = project_root / paths["taxonomy"]

    collection_df = load_collection_sheet(collection_csv)

    validate_collection_sheet(
        collection_df=collection_df,
        project_root=project_root,
        allowed_usage=list(rules["allowed_usage"]),
        require_image_files_exist=bool(rules["require_image_files_exist"]),
    )

    manifest = build_local_unknown_manifest_dataframe(
        collection_df=collection_df,
        rules=rules,
    )

    validate_manifest(manifest, taxonomy_path)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local unknown OpenWaste-HR manifest."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/local_unknown_manifest.yaml",
        help="Path to local unknown manifest YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    manifest = build_local_unknown_manifest(
        config_path=args.config,
        project_root=args.project_root,
    )

    print("Local unknown manifest created successfully.")
    print(f"Total samples: {len(manifest)}")
    print("\nUsage counts:")
    print(manifest["usage"].value_counts().to_string())
    print("\nOutput labels:")
    print(manifest["fine_label"].value_counts().to_string())


if __name__ == "__main__":
    main()