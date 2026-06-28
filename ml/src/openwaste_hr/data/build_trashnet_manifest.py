from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from openwaste_hr.data.manifest import validate_manifest


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_TRASHNET_CLASS_FOLDERS = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

TRASHNET_LABEL_MAPPING = {
    "cardboard": {
        "fine_label": "cardboard",
        "coarse_label": "recyclable",
    },
    "paper": {
        "fine_label": "paper",
        "coarse_label": "recyclable",
    },
    "plastic": {
        "fine_label": "plastic",
        "coarse_label": "recyclable",
    },
    "glass": {
        "fine_label": "glass",
        "coarse_label": "recyclable",
    },
    "metal": {
        "fine_label": "metal",
        "coarse_label": "recyclable",
    },
}


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


def make_relative_path(path: Path, project_root: Path) -> str:
    """
    Convert an absolute path into a project-relative path.
    """
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def discover_trashnet_images(dataset_root: str | Path, project_root: str | Path) -> pd.DataFrame:
    """
    Discover TrashNet images and map them into the clean 5-class taxonomy.

    The original TrashNet "trash" folder is expected to exist, but it is
    intentionally excluded from the main protocol because it is a mixed class.
    """
    dataset_root = Path(dataset_root)
    project_root = Path(project_root)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"TrashNet dataset root not found: {dataset_root}. "
            "Expected folders: cardboard, glass, metal, paper, plastic, trash."
        )

    rows: list[dict[str, Any]] = []
    sample_counter = 1

    for original_label in EXPECTED_TRASHNET_CLASS_FOLDERS:
        class_dir = dataset_root / original_label

        if not class_dir.exists():
            raise FileNotFoundError(f"Expected TrashNet class folder not found: {class_dir}")

    for original_label in sorted(TRASHNET_LABEL_MAPPING.keys()):
        class_dir = dataset_root / original_label

        image_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        )

        if not image_paths:
            raise ValueError(f"No supported images found in folder: {class_dir}")

        mapping = TRASHNET_LABEL_MAPPING[original_label]

        for image_path in image_paths:
            rows.append(
                {
                    "sample_id": f"trashnet_{sample_counter:06d}",
                    "source_dataset": "trashnet",
                    "source_split": "unassigned",
                    "image_path": make_relative_path(image_path, project_root),
                    "original_label": original_label,
                    "fine_label": mapping["fine_label"],
                    "coarse_label": mapping["coarse_label"],
                    "is_known": "true",
                    "usage": "unassigned",
                    "license_notes": "TrashNet dataset. Cite original TrashNet repository.",
                }
            )
            sample_counter += 1

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise ValueError("No TrashNet images were discovered.")

    return manifest


def split_manifest(
    manifest: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    Stratified split by original TrashNet label.

    This keeps each TrashNet class represented across train, validation, and test.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    split_parts: list[pd.DataFrame] = []

    for original_label, group in manifest.groupby("original_label", sort=True):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_total = len(shuffled)

        if n_total < 3:
            raise ValueError(
                f"Class '{original_label}' has only {n_total} images. "
                "At least 3 images are needed for train/val/test split."
            )

        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))

        if n_train + n_val >= n_total:
            n_train = max(1, n_total - 2)
            n_val = 1

        train_end = n_train
        val_end = n_train + n_val

        train_group = shuffled.iloc[:train_end].copy()
        val_group = shuffled.iloc[train_end:val_end].copy()
        test_group = shuffled.iloc[val_end:].copy()

        train_group["source_split"] = "train"
        train_group["usage"] = "known_train"

        val_group["source_split"] = "val"
        val_group["usage"] = "known_val"

        test_group["source_split"] = "test"
        test_group["usage"] = "known_test"

        split_parts.extend([train_group, val_group, test_group])

    split_manifest_df = pd.concat(split_parts, ignore_index=True)
    split_manifest_df = split_manifest_df.sort_values("sample_id").reset_index(drop=True)

    return split_manifest_df


def save_split_files(
    manifest: pd.DataFrame,
    output_manifest: str | Path,
    output_train: str | Path,
    output_val: str | Path,
    output_test: str | Path,
) -> None:
    """
    Save full manifest and individual split CSV files.
    """
    output_manifest = Path(output_manifest)
    output_train = Path(output_train)
    output_val = Path(output_val)
    output_test = Path(output_test)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(output_manifest, index=False)

    manifest[manifest["usage"] == "known_train"].to_csv(output_train, index=False)
    manifest[manifest["usage"] == "known_val"].to_csv(output_val, index=False)
    manifest[manifest["usage"] == "known_test"].to_csv(output_test, index=False)


def build_trashnet_manifest(config_path: str | Path, project_root: str | Path) -> pd.DataFrame:
    """
    Build the TrashNet manifest from config and validate it.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    split_config = config["split"]

    dataset_root = project_root / paths["dataset_root"]
    taxonomy_path = project_root / paths["taxonomy"]

    manifest = discover_trashnet_images(dataset_root, project_root)

    manifest = split_manifest(
        manifest=manifest,
        train_ratio=float(split_config["train_ratio"]),
        val_ratio=float(split_config["val_ratio"]),
        test_ratio=float(split_config["test_ratio"]),
        seed=int(split_config["seed"]),
    )

    validate_manifest(manifest, taxonomy_path)

    save_split_files(
        manifest=manifest,
        output_manifest=project_root / paths["output_manifest"],
        output_train=project_root / paths["output_train"],
        output_val=project_root / paths["output_val"],
        output_test=project_root / paths["output_test"],
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OpenWaste-HR manifest files from TrashNet folder structure."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/trashnet.yaml",
        help="Path to TrashNet intake YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    manifest = build_trashnet_manifest(
        config_path=args.config,
        project_root=args.project_root,
    )

    print("TrashNet manifest created successfully.")
    print(f"Total samples: {len(manifest)}")
    print("\nUsage counts:")
    print(manifest["usage"].value_counts().to_string())
    print("\nFine-label counts:")
    print(manifest["fine_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
