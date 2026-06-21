from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the RealWaste manifest for OpenWaste-HR."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory. Default: current directory.",
    )
    parser.add_argument(
        "--manifest",
        default="ml/data/splits/realwaste_manifest_v1.csv",
        help="Path to RealWaste manifest CSV.",
    )
    parser.add_argument(
        "--sample-read-limit",
        type=int,
        default=100,
        help="Maximum number of image files to open for readability checks.",
    )
    return parser.parse_args()


def value_counts_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, "count"]
    counts["proportion"] = counts["count"] / len(df)
    return counts


def grouped_counts_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    grouped = df.groupby(columns, dropna=False).size().reset_index(name="count")
    grouped["proportion"] = grouped["count"] / len(df)
    return grouped.sort_values("count", ascending=False)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """
    Convert a dataframe to a simple markdown table without requiring pandas.to_markdown.

    This avoids the optional pandas dependency on the external 'tabulate' package.
    """
    if df.empty:
        return "_No rows available._"

    headers = [str(column) for column in df.columns]

    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = []
        for column in df.columns:
            value = row[column]

            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def check_image_paths(
    df: pd.DataFrame,
    *,
    project_root: Path,
    sample_read_limit: int,
) -> dict[str, Any]:
    missing_paths: list[str] = []
    unreadable_paths: list[str] = []
    readable_count = 0
    checked_count = 0
    width_values: list[int] = []
    height_values: list[int] = []

    for image_path in df["image_path"].astype(str):
        full_path = project_root / image_path

        if not full_path.exists():
            missing_paths.append(image_path)
            continue

        if checked_count >= sample_read_limit:
            continue

        checked_count += 1

        try:
            with Image.open(full_path) as image:
                width, height = image.size
                width_values.append(int(width))
                height_values.append(int(height))
                readable_count += 1
        except Exception:
            unreadable_paths.append(image_path)

    dimension_summary = {
        "checked_image_count": checked_count,
        "readable_image_count": readable_count,
        "unreadable_image_count": len(unreadable_paths),
        "min_width": min(width_values) if width_values else None,
        "max_width": max(width_values) if width_values else None,
        "min_height": min(height_values) if height_values else None,
        "max_height": max(height_values) if height_values else None,
    }

    return {
        "missing_path_count": len(missing_paths),
        "missing_paths_preview": missing_paths[:10],
        "unreadable_path_count": len(unreadable_paths),
        "unreadable_paths_preview": unreadable_paths[:10],
        "dimension_summary": dimension_summary,
    }


def write_markdown_report(
    report_path: Path,
    *,
    summary: dict[str, Any],
    usage_counts: pd.DataFrame,
    fine_counts: pd.DataFrame,
    coarse_counts: pd.DataFrame,
    mapping_role_counts: pd.DataFrame,
    original_label_counts: pd.DataFrame,
) -> None:
    report = f"""# RealWaste Inspection v1 Report

## Summary

| metric | value |
| --- | ---: |
| total_samples | {summary["total_samples"]} |
| known_samples | {summary["known_samples"]} |
| unknown_samples | {summary["unknown_samples"]} |
| missing_path_count | {summary["image_file_check"]["missing_path_count"]} |
| unreadable_path_count | {summary["image_file_check"]["unreadable_path_count"]} |

## Usage Distribution

{dataframe_to_markdown_table(usage_counts)}

## Fine-Label Distribution

{dataframe_to_markdown_table(fine_counts)}

## Coarse-Label Distribution

{dataframe_to_markdown_table(coarse_counts)}

## Mapping-Role Distribution

{dataframe_to_markdown_table(mapping_role_counts)}

## Original Label Distribution

{dataframe_to_markdown_table(original_label_counts)}

## Image Dimension Sample Summary

| field | value |
| --- | ---: |
| checked_image_count | {summary["image_file_check"]["dimension_summary"]["checked_image_count"]} |
| readable_image_count | {summary["image_file_check"]["dimension_summary"]["readable_image_count"]} |
| unreadable_image_count | {summary["image_file_check"]["dimension_summary"]["unreadable_image_count"]} |
| min_width | {summary["image_file_check"]["dimension_summary"]["min_width"]} |
| max_width | {summary["image_file_check"]["dimension_summary"]["max_width"]} |
| min_height | {summary["image_file_check"]["dimension_summary"]["min_height"]} |
| max_height | {summary["image_file_check"]["dimension_summary"]["max_height"]} |

## Key Interpretation

RealWaste contributes expanded known-class data and a public-dataset unknown/future-class split.

The unknown split comes from Textile Trash, which is intentionally kept outside the current known taxonomy.

This supports the OpenWaste-HR open-set design and prepares the project for expanded training.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    manifest_path = project_root / args.manifest

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_metrics_dir = project_root / "ml" / "outputs" / "metrics"
    output_report_dir = project_root / "docs" / "results"

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    required_columns = {
        "sample_id",
        "source_dataset",
        "image_path",
        "original_label",
        "fine_label",
        "coarse_label",
        "is_known",
        "usage",
        "mapping_role",
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_columns)}")

    usage_counts = value_counts_table(df, "usage")
    fine_counts = value_counts_table(df, "fine_label")
    coarse_counts = value_counts_table(df, "coarse_label")
    mapping_role_counts = value_counts_table(df, "mapping_role")
    original_label_counts = value_counts_table(df, "original_label")
    label_mapping_counts = grouped_counts_table(
        df,
        [
            "original_label",
            "fine_label",
            "coarse_label",
            "is_known",
            "usage",
            "mapping_role",
        ],
    )

    image_file_check = check_image_paths(
        df,
        project_root=project_root,
        sample_read_limit=args.sample_read_limit,
    )

    summary = {
        "total_samples": int(len(df)),
        "known_samples": int(df["is_known"].astype(bool).sum()),
        "unknown_samples": int((~df["is_known"].astype(bool)).sum()),
        "usage_counts": {
            str(row["usage"]): int(row["count"])
            for _, row in usage_counts.iterrows()
        },
        "fine_label_counts": {
            str(row["fine_label"]): int(row["count"])
            for _, row in fine_counts.iterrows()
        },
        "coarse_label_counts": {
            str(row["coarse_label"]): int(row["count"])
            for _, row in coarse_counts.iterrows()
        },
        "mapping_role_counts": {
            str(row["mapping_role"]): int(row["count"])
            for _, row in mapping_role_counts.iterrows()
        },
        "image_file_check": image_file_check,
    }

    usage_counts_path = output_metrics_dir / "realwaste_usage_counts_v1.csv"
    fine_counts_path = output_metrics_dir / "realwaste_fine_label_counts_v1.csv"
    coarse_counts_path = output_metrics_dir / "realwaste_coarse_label_counts_v1.csv"
    mapping_role_counts_path = output_metrics_dir / "realwaste_mapping_role_counts_v1.csv"
    original_label_counts_path = output_metrics_dir / "realwaste_original_label_counts_v1.csv"
    label_mapping_counts_path = output_metrics_dir / "realwaste_label_mapping_counts_v1.csv"
    summary_path = output_metrics_dir / "realwaste_inspection_summary_v1.json"
    report_path = output_report_dir / "realwaste_inspection_v1_report.md"

    usage_counts.to_csv(usage_counts_path, index=False)
    fine_counts.to_csv(fine_counts_path, index=False)
    coarse_counts.to_csv(coarse_counts_path, index=False)
    mapping_role_counts.to_csv(mapping_role_counts_path, index=False)
    original_label_counts.to_csv(original_label_counts_path, index=False)
    label_mapping_counts.to_csv(label_mapping_counts_path, index=False)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_markdown_report(
        report_path,
        summary=summary,
        usage_counts=usage_counts,
        fine_counts=fine_counts,
        coarse_counts=coarse_counts,
        mapping_role_counts=mapping_role_counts,
        original_label_counts=original_label_counts,
    )

    print("RealWaste inspection completed successfully.")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Known samples: {summary['known_samples']}")
    print(f"Unknown/future-class samples: {summary['unknown_samples']}")
    print(f"Missing image paths: {image_file_check['missing_path_count']}")
    print(f"Unreadable checked images: {image_file_check['unreadable_path_count']}")
    print()
    print("Created files:")
    print(f"- usage counts: {usage_counts_path.relative_to(project_root)}")
    print(f"- fine-label counts: {fine_counts_path.relative_to(project_root)}")
    print(f"- coarse-label counts: {coarse_counts_path.relative_to(project_root)}")
    print(f"- mapping-role counts: {mapping_role_counts_path.relative_to(project_root)}")
    print(f"- original-label counts: {original_label_counts_path.relative_to(project_root)}")
    print(f"- label mapping counts: {label_mapping_counts_path.relative_to(project_root)}")
    print(f"- summary JSON: {summary_path.relative_to(project_root)}")
    print(f"- thesis report: {report_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()