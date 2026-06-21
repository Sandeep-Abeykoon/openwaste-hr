from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RealWaste manifest for OpenWaste-HR.")
    parser.add_argument(
        "--config",
        default="ml/configs/realwaste.yaml",
        help="Path to RealWaste config YAML.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )
    return parser.parse_args()


def normalise_label(label: str) -> str:
    cleaned = label.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_mapping(mapping_path: Path) -> pd.DataFrame:
    mapping_df = pd.read_csv(mapping_path)

    required_columns = {
        "original_label",
        "normalized_label",
        "fine_label",
        "coarse_label",
        "is_known",
        "mapping_role",
        "usage_if_included",
        "include_in_manifest",
        "license_notes",
        "mapping_notes",
    }

    missing = required_columns.difference(mapping_df.columns)
    if missing:
        raise ValueError(f"Mapping file is missing columns: {sorted(missing)}")

    mapping_df["normalized_label"] = mapping_df["normalized_label"].astype(str)
    return mapping_df


def find_dataset_root(raw_root: Path) -> Path:
    if not raw_root.exists():
        raise FileNotFoundError(
            f"RealWaste raw root not found: {raw_root}. "
            "Download/extract RealWaste first, then place it under ml/data/raw/realwaste."
        )

    direct_class_dirs = [path for path in raw_root.iterdir() if path.is_dir()]
    if direct_class_dirs:
        return raw_root

    nested = raw_root / "RealWaste"
    if nested.exists() and nested.is_dir():
        return nested

    return raw_root


def list_image_files(folder: Path, image_extensions: set[str]) -> list[Path]:
    files: list[Path] = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in image_extensions:
            files.append(path)

    return sorted(files)


def split_known_rows(
    known_rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rows = known_rows.copy()
    random.Random(random_seed).shuffle(rows)

    total = len(rows)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    for index, row in enumerate(rows):
        if index < train_end:
            row["usage"] = "known_train"
            row["source_split"] = "train"
        elif index < val_end:
            row["usage"] = "known_val"
            row["source_split"] = "val"
        else:
            row["usage"] = "known_test"
            row["source_split"] = "test"

    return rows


def build_manifest_rows(
    *,
    dataset_root: Path,
    project_root: Path,
    mapping_df: pd.DataFrame,
    image_extensions: set[str],
    source_dataset: str,
    default_license_notes: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mapping_by_normalized = {
        str(row["normalized_label"]): row.to_dict()
        for _, row in mapping_df.iterrows()
    }

    rows: list[dict[str, Any]] = []
    skipped_counts: dict[str, int] = {}

    class_dirs = sorted([path for path in dataset_root.iterdir() if path.is_dir()])

    for class_dir in class_dirs:
        original_label = class_dir.name
        normalized_label = normalise_label(original_label)

        mapping = mapping_by_normalized.get(normalized_label)

        if mapping is None:
            skipped_counts[original_label] = skipped_counts.get(original_label, 0) + 1
            continue

        include = str(mapping["include_in_manifest"]).strip().lower() == "true"
        if not include:
            skipped_counts[original_label] = skipped_counts.get(original_label, 0) + 1
            continue

        image_files = list_image_files(class_dir, image_extensions)

        for image_index, image_path in enumerate(image_files, start=1):
            sample_id = f"realwaste_{normalized_label}_{image_index:06d}"
            is_known = str(mapping["is_known"]).strip().lower() == "true"

            if is_known:
                usage = "known_train"
                source_split = "pending_split"
            else:
                usage = "unknown_test"
                source_split = "unknown_test"

            rows.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": source_dataset,
                    "source_split": source_split,
                    "image_path": str(image_path.relative_to(project_root)).replace("\\", "/"),
                    "original_label": original_label,
                    "fine_label": mapping["fine_label"],
                    "coarse_label": mapping["coarse_label"],
                    "is_known": is_known,
                    "usage": usage,
                    "license_notes": mapping.get("license_notes", default_license_notes),
                    "mapping_role": mapping["mapping_role"],
                    "mapping_notes": mapping["mapping_notes"],
                }
            )

    return rows, skipped_counts


def write_summary(
    *,
    summary_path: Path,
    manifest_df: pd.DataFrame,
    skipped_counts: dict[str, int],
    outputs: dict[str, str],
) -> None:
    if manifest_df.empty:
        summary = {
            "total_samples": 0,
            "known_samples": 0,
            "unknown_samples": 0,
            "usage_counts": {},
            "fine_label_counts": {},
            "coarse_label_counts": {},
            "mapping_role_counts": {},
            "skipped_counts": skipped_counts,
            "outputs": outputs,
        }
    else:
        summary = {
            "total_samples": int(len(manifest_df)),
            "known_samples": int(manifest_df["is_known"].sum()),
            "unknown_samples": int((~manifest_df["is_known"]).sum()),
            "usage_counts": {
                str(key): int(value)
                for key, value in manifest_df["usage"].value_counts().items()
            },
            "fine_label_counts": {
                str(key): int(value)
                for key, value in manifest_df["fine_label"].value_counts().items()
            },
            "coarse_label_counts": {
                str(key): int(value)
                for key, value in manifest_df["coarse_label"].value_counts().items()
            },
            "mapping_role_counts": {
                str(key): int(value)
                for key, value in manifest_df["mapping_role"].value_counts().items()
            },
            "skipped_counts": skipped_counts,
            "outputs": outputs,
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_realwaste_manifest(
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    config = load_config(config_path)

    raw_root = project_root / config["paths"]["raw_root"]
    mapping_path = project_root / config["paths"]["label_mapping_csv"]
    output_dir = project_root / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = find_dataset_root(raw_root)
    mapping_df = load_mapping(mapping_path)

    image_extensions = {extension.lower() for extension in config["image_extensions"]}

    rows, skipped_counts = build_manifest_rows(
        dataset_root=dataset_root,
        project_root=project_root,
        mapping_df=mapping_df,
        image_extensions=image_extensions,
        source_dataset=config["source_dataset"]["name"],
        default_license_notes=config["source_dataset"]["license_notes"],
    )

    if not rows:
        raise ValueError(
            f"No RealWaste images were found under {dataset_root}. "
            "Check the dataset folder structure and image extensions."
        )

    known_rows = [row for row in rows if row["is_known"]]
    unknown_rows = [row for row in rows if not row["is_known"]]

    split_config = config["split"]
    known_rows = split_known_rows(
        known_rows,
        train_ratio=float(split_config["train_ratio"]),
        val_ratio=float(split_config["val_ratio"]),
        test_ratio=float(split_config["test_ratio"]),
        random_seed=int(split_config["random_seed"]),
    )

    final_rows = known_rows + unknown_rows
    manifest_df = pd.DataFrame(final_rows).sort_values("sample_id")

    manifest_path = output_dir / config["outputs"]["manifest_csv"]
    known_train_path = output_dir / config["outputs"]["known_train_csv"]
    known_val_path = output_dir / config["outputs"]["known_val_csv"]
    known_test_path = output_dir / config["outputs"]["known_test_csv"]
    unknown_test_path = output_dir / config["outputs"]["unknown_test_csv"]
    summary_path = output_dir / config["outputs"]["summary_json"]

    manifest_df.to_csv(manifest_path, index=False)
    manifest_df[manifest_df["usage"] == "known_train"].to_csv(known_train_path, index=False)
    manifest_df[manifest_df["usage"] == "known_val"].to_csv(known_val_path, index=False)
    manifest_df[manifest_df["usage"] == "known_test"].to_csv(known_test_path, index=False)
    manifest_df[manifest_df["usage"] == "unknown_test"].to_csv(unknown_test_path, index=False)

    outputs = {
        "manifest_csv": str(manifest_path.relative_to(project_root)),
        "known_train_csv": str(known_train_path.relative_to(project_root)),
        "known_val_csv": str(known_val_path.relative_to(project_root)),
        "known_test_csv": str(known_test_path.relative_to(project_root)),
        "unknown_test_csv": str(unknown_test_path.relative_to(project_root)),
        "summary_json": str(summary_path.relative_to(project_root)),
    }

    write_summary(
        summary_path=summary_path,
        manifest_df=manifest_df,
        skipped_counts=skipped_counts,
        outputs=outputs,
    )

    return {
        "total_samples": int(len(manifest_df)),
        "known_samples": int(manifest_df["is_known"].sum()),
        "unknown_samples": int((~manifest_df["is_known"]).sum()),
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = project_root / args.config

    result = build_realwaste_manifest(
        config_path=config_path,
        project_root=project_root,
    )

    print("RealWaste manifest created successfully.")
    print(f"Total samples: {result['total_samples']}")
    print(f"Known samples: {result['known_samples']}")
    print(f"Unknown/future-class samples: {result['unknown_samples']}")
    print()
    print("Created files:")
    for label, path in result["outputs"].items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()