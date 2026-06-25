from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError

from openwaste_hr.data.manifest import load_manifest, validate_manifest


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


def resolve_project_path(project_root: str | Path, relative_path: str | Path) -> Path:
    """
    Resolve a project-relative path.
    """
    return Path(project_root) / Path(relative_path)


def inspect_image_files(manifest: pd.DataFrame, project_root: str | Path) -> pd.DataFrame:
    """
    Check whether each image exists and can be opened by PIL.
    """
    project_root = Path(project_root)

    rows: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        image_path_text = str(row["image_path"])
        image_path = project_root / image_path_text

        exists = image_path.exists()
        readable = False
        width = None
        height = None
        error = ""

        if exists:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify()
                readable = True
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                error = str(exc)
        else:
            error = "File does not exist."

        rows.append(
            {
                "sample_id": row["sample_id"],
                "image_path": image_path_text,
                "exists": exists,
                "readable": readable,
                "width": width,
                "height": height,
                "error": error,
            }
        )

    return pd.DataFrame(rows)


def fail_if_invalid_images(image_report: pd.DataFrame) -> None:
    """
    Raise an error if missing or unreadable images are found.
    """
    invalid_rows = image_report[
        (image_report["exists"] == False) | (image_report["readable"] == False)  # noqa: E712
    ]

    if not invalid_rows.empty:
        preview = invalid_rows.head(10)[["sample_id", "image_path", "error"]]
        raise ValueError(
            "Missing or unreadable images found. First invalid rows:\n"
            f"{preview.to_string(index=False)}"
        )


def build_count_table(manifest: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Build count and percentage table for a manifest column.
    """
    if column not in manifest.columns:
        raise ValueError(f"Column not found in manifest: {column}")

    counts = (
        manifest[column]
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )

    counts["percentage"] = (counts["count"] / len(manifest) * 100).round(2)

    return counts


def build_dimension_summary(image_report: pd.DataFrame) -> pd.DataFrame:
    """
    Build width and height summary for readable images.
    """
    readable = image_report[image_report["readable"] == True].copy()  # noqa: E712

    if readable.empty:
        raise ValueError("No readable images available for dimension summary.")

    width_values = readable["width"].astype(float)
    height_values = readable["height"].astype(float)

    rows = [
        {
            "metric": "total_manifest_rows",
            "value": len(image_report),
        },
        {
            "metric": "readable_images",
            "value": len(readable),
        },
        {
            "metric": "min_width",
            "value": int(width_values.min()),
        },
        {
            "metric": "max_width",
            "value": int(width_values.max()),
        },
        {
            "metric": "mean_width",
            "value": round(float(width_values.mean()), 2),
        },
        {
            "metric": "min_height",
            "value": int(height_values.min()),
        },
        {
            "metric": "max_height",
            "value": int(height_values.max()),
        },
        {
            "metric": "mean_height",
            "value": round(float(height_values.mean()), 2),
        },
    ]

    return pd.DataFrame(rows)


def save_count_tables(
    manifest: pd.DataFrame,
    output_metrics_dir: Path,
    output_names: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """
    Save count tables for usage, fine label, coarse label, and original label.
    """
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "usage_counts": build_count_table(manifest, "usage"),
        "fine_label_counts": build_count_table(manifest, "fine_label"),
        "coarse_label_counts": build_count_table(manifest, "coarse_label"),
        "original_label_counts": build_count_table(manifest, "original_label"),
    }

    tables["usage_counts"].to_csv(
        output_metrics_dir / output_names["usage_counts"],
        index=False,
    )
    tables["fine_label_counts"].to_csv(
        output_metrics_dir / output_names["fine_label_counts"],
        index=False,
    )
    tables["coarse_label_counts"].to_csv(
        output_metrics_dir / output_names["coarse_label_counts"],
        index=False,
    )
    tables["original_label_counts"].to_csv(
        output_metrics_dir / output_names["original_label_counts"],
        index=False,
    )

    return tables


def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a simple Markdown table without requiring tabulate.
    """
    if dataframe.empty:
        return "_No rows._"

    columns = list(dataframe.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    body_lines = []
    for _, row in dataframe.iterrows():
        values = [str(row[column]) for column in columns]
        body_lines.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *body_lines])


def write_markdown_report(
    manifest: pd.DataFrame,
    image_report: pd.DataFrame,
    count_tables: dict[str, pd.DataFrame],
    dimension_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write a small inspection report for local documentation.
    """
    invalid_count = len(
        image_report[
            (image_report["exists"] == False) | (image_report["readable"] == False)  # noqa: E712
        ]
    )

    report = f"""# TrashNet Data Inspection Report v1

## Summary

| Metric | Value |
|---|---:|
| Total manifest rows | {len(manifest)} |
| Missing or unreadable images | {invalid_count} |
| Unique fine labels | {manifest["fine_label"].nunique()} |
| Unique coarse labels | {manifest["coarse_label"].nunique()} |
| Unique original labels | {manifest["original_label"].nunique()} |

## Usage Counts

{dataframe_to_markdown_table(count_tables["usage_counts"])}

## Fine Label Counts

{dataframe_to_markdown_table(count_tables["fine_label_counts"])}

## Coarse Label Counts

{dataframe_to_markdown_table(count_tables["coarse_label_counts"])}

## Original Label Counts

{dataframe_to_markdown_table(count_tables["original_label_counts"])}

## Image Dimension Summary

{dataframe_to_markdown_table(dimension_summary)}

## Research Observation

This inspection should be used to record dataset limitations before model training.

For TrashNet v1, the dataset is useful for the first closed-set baseline, but it does not fully cover the OpenWaste-HR target taxonomy. In particular, TrashNet does not provide dedicated organic or e-waste/battery classes. Therefore, later stages must add additional public or local data for full taxonomy coverage and open-world evaluation.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def plot_fine_label_distribution(
    count_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save a bar chart showing fine-label distribution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = count_table.sort_values("count", ascending=False)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(plot_data["fine_label"], plot_data["count"])
    axis.set_title("TrashNet Fine Label Distribution")
    axis.set_xlabel("Fine Label")
    axis.set_ylabel("Image Count")
    axis.tick_params(axis="x", rotation=35)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def create_sample_grid(
    manifest: pd.DataFrame,
    project_root: str | Path,
    output_path: Path,
    images_per_label: int,
    seed: int,
) -> None:
    """
    Save a grid of sample images from each fine label.
    """
    project_root = Path(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sampled_parts = []

    for fine_label, group in manifest.groupby("fine_label", sort=True):
        sample_count = min(images_per_label, len(group))
        sampled = group.sample(n=sample_count, random_state=seed).copy()
        sampled_parts.append(sampled)

    sampled_manifest = pd.concat(sampled_parts, ignore_index=True)

    total_images = len(sampled_manifest)
    columns = 4
    rows = math.ceil(total_images / columns)

    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 3))

    if rows == 1:
        axes_list = list(axes)
    else:
        axes_list = [axis for row_axes in axes for axis in row_axes]

    for axis in axes_list:
        axis.axis("off")

    for axis, (_, row) in zip(axes_list, sampled_manifest.iterrows()):
        image_path = project_root / str(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            axis.imshow(image)

        axis.set_title(f"{row['fine_label']}\n{row['original_label']}", fontsize=9)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def inspect_dataset(config_path: str | Path, project_root: str | Path) -> dict[str, Path]:
    """
    Run the full dataset inspection process.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    output_names = config["outputs"]
    rules = config["rules"]
    sampling = config["sampling"]

    manifest_path = resolve_project_path(project_root, paths["manifest"])
    taxonomy_path = resolve_project_path(project_root, paths["taxonomy"])
    output_metrics_dir = resolve_project_path(project_root, paths["output_metrics_dir"])
    output_figures_dir = resolve_project_path(project_root, paths["output_figures_dir"])

    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, taxonomy_path)

    image_report = inspect_image_files(manifest, project_root)

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    image_report_path = output_metrics_dir / output_names["image_file_report"]
    image_report.to_csv(image_report_path, index=False)

    if rules.get("fail_on_missing_images", True) or rules.get("fail_on_unreadable_images", True):
        fail_if_invalid_images(image_report)

    count_tables = save_count_tables(
        manifest=manifest,
        output_metrics_dir=output_metrics_dir,
        output_names=output_names,
    )

    dimension_summary = build_dimension_summary(image_report)
    dimension_summary_path = output_metrics_dir / output_names["dimension_summary"]
    dimension_summary.to_csv(dimension_summary_path, index=False)

    markdown_report_path = output_metrics_dir / output_names["markdown_report"]
    write_markdown_report(
        manifest=manifest,
        image_report=image_report,
        count_tables=count_tables,
        dimension_summary=dimension_summary,
        output_path=markdown_report_path,
    )

    fine_label_plot_path = output_figures_dir / output_names["fine_label_distribution_plot"]
    plot_fine_label_distribution(
        count_table=count_tables["fine_label_counts"],
        output_path=fine_label_plot_path,
    )

    sample_grid_path = output_figures_dir / output_names["sample_grid"]
    create_sample_grid(
        manifest=manifest,
        project_root=project_root,
        output_path=sample_grid_path,
        images_per_label=int(sampling["sample_grid_images_per_label"]),
        seed=int(sampling["sample_seed"]),
    )

    return {
        "image_file_report": image_report_path,
        "dimension_summary": dimension_summary_path,
        "markdown_report": markdown_report_path,
        "fine_label_distribution_plot": fine_label_plot_path,
        "sample_grid": sample_grid_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect OpenWaste-HR dataset manifest and image files."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/data_inspection.yaml",
        help="Path to data inspection YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    output_paths = inspect_dataset(
        config_path=args.config,
        project_root=args.project_root,
    )

    print("Dataset inspection completed successfully.")
    print("\nCreated files:")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()