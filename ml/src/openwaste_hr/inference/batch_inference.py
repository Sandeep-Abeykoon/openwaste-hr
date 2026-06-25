from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from openwaste_hr.evaluation.hierarchical_decision import apply_hierarchical_policy
from openwaste_hr.inference.single_image_inference import (
    build_prediction_dataframe,
    get_device,
    load_image_tensor,
    load_model_for_inference,
    load_yaml,
    resolve_class_names,
    resolve_path,
    run_model_prediction,
)


def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No rows._"

    columns = list(dataframe.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in dataframe.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def list_image_files(
    image_dir: str | Path,
    image_globs: list[str],
    max_images: int | None = None,
) -> list[Path]:
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {image_dir}")

    if not image_globs:
        raise ValueError("At least one image glob pattern is required.")

    if max_images is not None and int(max_images) <= 0:
        raise ValueError("max_images must be positive or null.")

    files_by_key: dict[str, Path] = {}

    for pattern in image_globs:
        for file_path in image_dir.glob(pattern):
            if file_path.is_file():
                files_by_key[str(file_path.resolve()).lower()] = file_path

    image_files = sorted(
        files_by_key.values(),
        key=lambda path: path.name.lower(),
    )

    if max_images is not None:
        image_files = image_files[: int(max_images)]

    if not image_files:
        raise ValueError(f"No image files found in: {image_dir}")

    return image_files


def make_project_relative_path(
    project_root: str | Path,
    file_path: str | Path,
) -> str:
    project_root = Path(project_root).resolve()
    file_path = Path(file_path).resolve()

    try:
        return file_path.relative_to(project_root).as_posix()
    except ValueError:
        return file_path.as_posix()


def run_batch_model_predictions(
    model: torch.nn.Module,
    image_paths: list[Path],
    project_root: str | Path,
    image_size: int,
    class_names: list[str],
    device: torch.device,
) -> pd.DataFrame:
    prediction_frames: list[pd.DataFrame] = []

    for image_path in image_paths:
        relative_image_path = make_project_relative_path(
            project_root=project_root,
            file_path=image_path,
        )

        image_tensor = load_image_tensor(
            image_path=image_path,
            image_size=image_size,
            device=device,
        )

        probabilities = run_model_prediction(
            model=model,
            image_tensor=image_tensor,
            class_names=class_names,
        )

        prediction_df = build_prediction_dataframe(
            sample_id=image_path.stem,
            image_path=relative_image_path,
            class_names=class_names,
            probabilities=probabilities,
        )

        prediction_frames.append(prediction_df)

    if not prediction_frames:
        raise ValueError("No predictions were generated.")

    return pd.concat(prediction_frames, ignore_index=True)


def build_decision_distribution(decisions_df: pd.DataFrame) -> pd.DataFrame:
    total_images = len(decisions_df)

    rows: list[dict[str, Any]] = []

    for decision_type in ["fine_label", "coarse_label", "manual_review"]:
        count = int(
            (decisions_df["hierarchical_decision_type"] == decision_type).sum()
        )

        rows.append(
            {
                "decision_type": decision_type,
                "count": count,
                "percentage": round(float(count / total_images * 100), 2)
                if total_images > 0
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_final_label_distribution(decisions_df: pd.DataFrame) -> pd.DataFrame:
    distribution_df = (
        decisions_df["hierarchical_final_label"]
        .value_counts()
        .rename_axis("final_label")
        .reset_index(name="count")
    )

    total_images = len(decisions_df)

    distribution_df["percentage"] = distribution_df["count"].apply(
        lambda count: round(float(count / total_images * 100), 2)
        if total_images > 0
        else 0.0
    )

    return distribution_df


def build_batch_summary(decisions_df: pd.DataFrame) -> dict[str, Any]:
    total_images = int(len(decisions_df))

    decision_counts = decisions_df["hierarchical_decision_type"].value_counts().to_dict()

    fine_label_count = int(decision_counts.get("fine_label", 0))
    coarse_label_count = int(decision_counts.get("coarse_label", 0))
    manual_review_count = int(decision_counts.get("manual_review", 0))
    accepted_count = fine_label_count + coarse_label_count

    return {
        "total_images": total_images,
        "fine_label_count": fine_label_count,
        "coarse_label_count": coarse_label_count,
        "manual_review_count": manual_review_count,
        "accepted_count": accepted_count,
        "fine_label_rate": round(float(fine_label_count / total_images), 6)
        if total_images > 0
        else 0.0,
        "coarse_label_rate": round(float(coarse_label_count / total_images), 6)
        if total_images > 0
        else 0.0,
        "manual_review_rate": round(float(manual_review_count / total_images), 6)
        if total_images > 0
        else 0.0,
        "accepted_rate": round(float(accepted_count / total_images), 6)
        if total_images > 0
        else 0.0,
    }


def write_markdown_report(
    output_path: Path,
    summary: dict[str, Any],
    decision_distribution_df: pd.DataFrame,
    final_label_distribution_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in summary.items()
        ]
    )

    preview_columns = [
        "sample_id",
        "image_path",
        "pred_label",
        "max_softmax_confidence",
        "top_coarse_label",
        "top_coarse_confidence",
        "coarse_margin",
        "hierarchical_decision_type",
        "hierarchical_final_label",
        "hierarchical_decision_reason",
    ]

    available_preview_columns = [
        column
        for column in preview_columns
        if column in decisions_df.columns
    ]

    preview_df = decisions_df[available_preview_columns].head(20).copy()

    report = f"""# Batch Inference Pipeline v1 Report

## Purpose

This report documents batch inference results for a folder of images.

## Summary

{dataframe_to_markdown_table(summary_df)}

## Decision Distribution

{dataframe_to_markdown_table(decision_distribution_df)}

## Final Label Distribution

{dataframe_to_markdown_table(final_label_distribution_df)}

## Result Preview

{dataframe_to_markdown_table(preview_df)}

## Research Interpretation

This stage demonstrates that OpenWaste-HR can process a folder of images and produce structured fine-label, coarse-label, or manual-review decisions.

The batch output can be used for analysis, manual review, active learning, or later backend/frontend integration.
"""

    output_path.write_text(report, encoding="utf-8")


def run_batch_inference(
    config_path: str | Path,
    project_root: str | Path,
    image_dir_override: str | None = None,
    max_images_override: int | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    config = load_yaml(config_path)

    model_config = config["model"]
    labels_config = config["labels"]
    policy_config = config["policy"]
    input_config = config["input"]
    outputs_config = config["outputs"]

    image_dir = image_dir_override or input_config["image_dir"]
    resolved_image_dir = resolve_path(project_root, image_dir)

    max_images = max_images_override

    if max_images is None:
        max_images = input_config.get("max_images", None)

    image_paths = list_image_files(
        image_dir=resolved_image_dir,
        image_globs=list(input_config["image_globs"]),
        max_images=max_images,
    )

    checkpoint_path = resolve_path(project_root, model_config["checkpoint_path"])
    output_metrics_dir = resolve_path(project_root, outputs_config["output_metrics_dir"])
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    model, checkpoint = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        model_name=str(model_config["model_name"]),
        num_classes=int(model_config["num_classes"]),
        pretrained=bool(model_config["pretrained"]),
        device=device,
    )

    class_names = resolve_class_names(
        checkpoint=checkpoint,
        config_class_names=list(model_config["class_names"]),
    )

    predictions_df = run_batch_model_predictions(
        model=model,
        image_paths=image_paths,
        project_root=project_root,
        image_size=int(model_config["image_size"]),
        class_names=class_names,
        device=device,
    )

    decisions_df = apply_hierarchical_policy(
        predictions_df=predictions_df,
        fine_to_coarse={
            str(fine_label): str(coarse_label)
            for fine_label, coarse_label in labels_config["fine_to_coarse"].items()
        },
        fine_confidence_threshold=float(policy_config["fine_confidence_threshold"]),
        coarse_confidence_threshold=float(policy_config["coarse_confidence_threshold"]),
        coarse_margin_threshold=float(policy_config["coarse_margin_threshold"]),
        minimum_confidence_for_coarse=float(
            policy_config["minimum_confidence_for_coarse"]
        ),
    )

    decision_distribution_df = build_decision_distribution(decisions_df)
    final_label_distribution_df = build_final_label_distribution(decisions_df)
    summary = build_batch_summary(decisions_df)

    results_path = output_metrics_dir / outputs_config["output_results_csv"]
    summary_path = output_metrics_dir / outputs_config["output_summary_json"]
    report_path = output_metrics_dir / outputs_config["output_markdown_report"]

    decisions_df.to_csv(results_path, index=False)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        output_path=report_path,
        summary=summary,
        decision_distribution_df=decision_distribution_df,
        final_label_distribution_df=final_label_distribution_df,
        decisions_df=decisions_df,
    )

    print("Batch inference completed successfully.")
    print(f"Images processed: {len(decisions_df)}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nCreated files:")
    print(f"- batch results: {results_path}")
    print(f"- summary: {summary_path}")
    print(f"- report: {report_path}")

    return {
        "summary": summary,
        "results_csv": str(results_path),
        "summary_json": str(summary_path),
        "report": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OpenWaste-HR batch inference on a folder of images."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/batch_inference_pipeline.yaml",
        help="Path to batch inference YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional project-relative image directory override.",
    )
    parser.add_argument(
        "--max-images",
        default=None,
        type=int,
        help="Optional maximum number of images to process.",
    )

    args = parser.parse_args()

    run_batch_inference(
        config_path=args.config,
        project_root=args.project_root,
        image_dir_override=args.image_dir,
        max_images_override=args.max_images,
    )


if __name__ == "__main__":
    main()