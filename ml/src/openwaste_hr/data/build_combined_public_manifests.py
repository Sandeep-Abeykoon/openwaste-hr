from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_COLUMNS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined TrashNet + RealWaste public training manifests."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/combined_public_training_manifests.yaml",
        help="Path to combined public manifest config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def read_manifest_csv(path: Path, *, expected_usage: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    if expected_usage is not None:
        unexpected = sorted(set(df["usage"].astype(str)) - {expected_usage})
        if unexpected:
            raise ValueError(
                f"{path} contains unexpected usage values {unexpected}; "
                f"expected only {expected_usage}."
            )

    return df


def normalise_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    if "mapping_role" not in output.columns:
        output["mapping_role"] = "known_train_candidate"

    if "mapping_notes" not in output.columns:
        output["mapping_notes"] = "not_provided"

    output["is_known"] = output["is_known"].astype(bool)
    output["sample_id"] = output["sample_id"].astype(str)
    output["source_dataset"] = output["source_dataset"].astype(str)
    output["usage"] = output["usage"].astype(str)

    return output


def assert_no_duplicate_sample_ids(df: pd.DataFrame) -> None:
    duplicates = df[df["sample_id"].duplicated()]["sample_id"].tolist()

    if duplicates:
        preview = duplicates[:10]
        raise ValueError(f"Duplicate sample_id values found: {preview}")


def validate_known_labels(
    *,
    train_df: pd.DataFrame,
    known_df: pd.DataFrame,
    allowed_known_fine_labels: set[str],
    required_known_train_labels: set[str],
) -> None:
    known_labels = set(known_df["fine_label"].astype(str))
    invalid_labels = known_labels - allowed_known_fine_labels

    if invalid_labels:
        raise ValueError(f"Invalid known fine labels found: {sorted(invalid_labels)}")

    train_labels = set(train_df["fine_label"].astype(str))
    missing_required = required_known_train_labels - train_labels

    if missing_required:
        raise ValueError(
            "Expanded known training split is missing required labels: "
            f"{sorted(missing_required)}"
        )


def validate_unknown_split(
    *,
    unknown_df: pd.DataFrame,
    unknown_fine_label: str,
    unknown_coarse_label: str,
) -> None:
    if unknown_df.empty:
        raise ValueError("Combined public unknown split is empty.")

    if set(unknown_df["fine_label"].astype(str)) != {unknown_fine_label}:
        raise ValueError("Unknown split must only contain fine_label='unknown'.")

    if set(unknown_df["coarse_label"].astype(str)) != {unknown_coarse_label}:
        raise ValueError("Unknown split must only contain coarse_label='unknown'.")

    if set(unknown_df["is_known"].astype(bool)) != {False}:
        raise ValueError("Unknown split must only contain is_known=False.")


def value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in df[column].value_counts().items()
    }


def build_combined_public_manifests(
    *,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    config = load_config(config_path)

    trashnet_config = config["inputs"]["trashnet"]
    realwaste_config = config["inputs"]["realwaste"]

    trashnet_train = read_manifest_csv(
        project_root / trashnet_config["known_train_csv"],
        expected_usage="known_train",
    )
    trashnet_val = read_manifest_csv(
        project_root / trashnet_config["known_val_csv"],
        expected_usage="known_val",
    )
    trashnet_test = read_manifest_csv(
        project_root / trashnet_config["known_test_csv"],
        expected_usage="known_test",
    )

    realwaste_train = read_manifest_csv(
        project_root / realwaste_config["known_train_csv"],
        expected_usage="known_train",
    )
    realwaste_val = read_manifest_csv(
        project_root / realwaste_config["known_val_csv"],
        expected_usage="known_val",
    )
    realwaste_test = read_manifest_csv(
        project_root / realwaste_config["known_test_csv"],
        expected_usage="known_test",
    )
    realwaste_unknown = read_manifest_csv(
        project_root / realwaste_config["unknown_test_csv"],
        expected_usage="unknown_test",
    )

    train_df = pd.concat(
        [normalise_manifest_columns(trashnet_train), normalise_manifest_columns(realwaste_train)],
        ignore_index=True,
    )
    val_df = pd.concat(
        [normalise_manifest_columns(trashnet_val), normalise_manifest_columns(realwaste_val)],
        ignore_index=True,
    )
    test_df = pd.concat(
        [normalise_manifest_columns(trashnet_test), normalise_manifest_columns(realwaste_test)],
        ignore_index=True,
    )
    unknown_df = normalise_manifest_columns(realwaste_unknown)

    train_df["usage"] = "known_train"
    val_df["usage"] = "known_val"
    test_df["usage"] = "known_test"
    unknown_df["usage"] = "unknown_test"

    known_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df = pd.concat([known_df, unknown_df], ignore_index=True)

    assert_no_duplicate_sample_ids(combined_df)

    validation_config = config["validation"]
    validate_known_labels(
        train_df=train_df,
        known_df=known_df,
        allowed_known_fine_labels=set(validation_config["allowed_known_fine_labels"]),
        required_known_train_labels=set(validation_config["require_known_train_labels"]),
    )
    validate_unknown_split(
        unknown_df=unknown_df,
        unknown_fine_label=str(validation_config["unknown_fine_label"]),
        unknown_coarse_label=str(validation_config["unknown_coarse_label"]),
    )

    outputs = config["outputs"]
    output_dir = project_root / outputs["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_manifest_path = output_dir / outputs["combined_manifest_csv"]
    combined_train_path = output_dir / outputs["combined_known_train_csv"]
    combined_val_path = output_dir / outputs["combined_known_val_csv"]
    combined_test_path = output_dir / outputs["combined_known_test_csv"]
    combined_unknown_path = output_dir / outputs["combined_unknown_test_csv"]
    summary_path = output_dir / outputs["summary_json"]

    combined_df = combined_df.sort_values(["usage", "source_dataset", "sample_id"])
    train_df = train_df.sort_values(["source_dataset", "sample_id"])
    val_df = val_df.sort_values(["source_dataset", "sample_id"])
    test_df = test_df.sort_values(["source_dataset", "sample_id"])
    unknown_df = unknown_df.sort_values(["source_dataset", "sample_id"])

    combined_df.to_csv(combined_manifest_path, index=False)
    train_df.to_csv(combined_train_path, index=False)
    val_df.to_csv(combined_val_path, index=False)
    test_df.to_csv(combined_test_path, index=False)
    unknown_df.to_csv(combined_unknown_path, index=False)

    summary = {
        "total_combined_samples": int(len(combined_df)),
        "known_samples": int(len(known_df)),
        "unknown_samples": int(len(unknown_df)),
        "known_train_samples": int(len(train_df)),
        "known_val_samples": int(len(val_df)),
        "known_test_samples": int(len(test_df)),
        "source_dataset_counts": value_counts_dict(combined_df, "source_dataset"),
        "usage_counts": value_counts_dict(combined_df, "usage"),
        "fine_label_counts": value_counts_dict(combined_df, "fine_label"),
        "coarse_label_counts": value_counts_dict(combined_df, "coarse_label"),
        "mapping_role_counts": value_counts_dict(combined_df, "mapping_role"),
        "outputs": {
            "combined_manifest_csv": str(combined_manifest_path.relative_to(project_root)),
            "combined_known_train_csv": str(combined_train_path.relative_to(project_root)),
            "combined_known_val_csv": str(combined_val_path.relative_to(project_root)),
            "combined_known_test_csv": str(combined_test_path.relative_to(project_root)),
            "combined_unknown_test_csv": str(combined_unknown_path.relative_to(project_root)),
            "summary_json": str(summary_path.relative_to(project_root)),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = project_root / args.config

    summary = build_combined_public_manifests(
        config_path=config_path,
        project_root=project_root,
    )

    print("Combined public training manifests created successfully.")
    print(f"Total combined samples: {summary['total_combined_samples']}")
    print(f"Known samples: {summary['known_samples']}")
    print(f"Unknown/future-class samples: {summary['unknown_samples']}")
    print(f"Known train samples: {summary['known_train_samples']}")
    print(f"Known validation samples: {summary['known_val_samples']}")
    print(f"Known test samples: {summary['known_test_samples']}")
    print()
    print("Fine-label counts:")
    for label, count in summary["fine_label_counts"].items():
        print(f"- {label}: {count}")
    print()
    print("Created files:")
    for key, path in summary["outputs"].items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()