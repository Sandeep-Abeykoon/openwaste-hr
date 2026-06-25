from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML dictionary.")

    return config


def resolve_path(project_root: str | Path, path_text: str | Path) -> Path:
    return Path(project_root) / Path(path_text)


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


def validate_candidate_columns(candidates_df: pd.DataFrame) -> bool:
    required_columns = {
        "candidate_rank",
        "sample_id",
        "image_path",
        "pred_label",
        "max_softmax_confidence",
        "hierarchical_decision_type",
        "hierarchical_final_label",
        "active_learning_score",
        "active_learning_reason",
    }

    missing_columns = required_columns - set(candidates_df.columns)

    if missing_columns:
        raise ValueError(
            f"Candidate file is missing required columns: {sorted(missing_columns)}"
        )

    return True


def build_human_labelling_sheet(
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a human labelling sheet from active learning candidates.
    """
    validate_candidate_columns(candidates_df)

    output_columns = [
        "candidate_rank",
        "sample_id",
        "image_path",
        "object_description",
        "why_unknown_or_difficult",
        "lighting_condition",
        "background_condition",
        "pred_label",
        "max_softmax_confidence",
        "top_coarse_label",
        "top_coarse_confidence",
        "coarse_margin",
        "hierarchical_decision_type",
        "hierarchical_final_label",
        "active_learning_score",
        "active_learning_reason",
        "recommended_human_action",
    ]

    available_columns = [
        column
        for column in output_columns
        if column in candidates_df.columns
    ]

    labelling_df = candidates_df[available_columns].copy()

    labelling_df["human_decision"] = ""
    labelling_df["human_fine_label"] = ""
    labelling_df["human_coarse_label"] = ""
    labelling_df["proposed_new_label"] = ""
    labelling_df["human_confidence"] = ""
    labelling_df["human_notes"] = ""
    labelling_df["reviewed_by"] = ""
    labelling_df["review_date"] = ""

    return labelling_df


def validate_human_labelling_sheet(
    labelling_df: pd.DataFrame,
) -> bool:
    required_annotation_columns = {
        "human_decision",
        "human_fine_label",
        "human_coarse_label",
        "proposed_new_label",
        "human_confidence",
        "human_notes",
        "reviewed_by",
        "review_date",
    }

    missing_columns = required_annotation_columns - set(labelling_df.columns)

    if missing_columns:
        raise ValueError(
            f"Human labelling sheet is missing columns: {sorted(missing_columns)}"
        )

    if labelling_df.empty:
        raise ValueError("Human labelling sheet cannot be empty.")

    if "sample_id" not in labelling_df.columns:
        raise ValueError("Human labelling sheet must contain sample_id.")

    if not labelling_df["sample_id"].is_unique:
        raise ValueError("Human labelling sheet sample_id values must be unique.")

    return True


def build_labelling_summary(
    labelling_df: pd.DataFrame,
) -> dict[str, Any]:
    decision_counts = (
        labelling_df["hierarchical_decision_type"]
        .value_counts()
        .to_dict()
        if "hierarchical_decision_type" in labelling_df.columns
        else {}
    )

    return {
        "total_labelling_rows": int(len(labelling_df)),
        "manual_review_candidates": int(decision_counts.get("manual_review", 0)),
        "coarse_label_candidates": int(decision_counts.get("coarse_label", 0)),
        "fine_label_candidates": int(decision_counts.get("fine_label", 0)),
    }


def write_human_labelling_instructions(
    output_path: Path,
    annotation_config: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_decisions = annotation_config["allowed_human_decisions"]
    allowed_fine_labels = annotation_config["allowed_fine_labels"]
    allowed_coarse_labels = annotation_config["allowed_coarse_labels"]
    allowed_confidence = annotation_config["allowed_human_confidence"]

    text = f"""# Human Labelling Instructions v1

## Purpose

Use this guide when filling `human_labelling_sheet_v1.csv`.

## Allowed Human Decisions

{chr(10).join(f"- {item}" for item in allowed_decisions)}

## Allowed Fine Labels

{chr(10).join(f"- {item}" for item in allowed_fine_labels)}

## Allowed Coarse Labels

{chr(10).join(f"- {item}" for item in allowed_coarse_labels)}

## Allowed Human Confidence Values

{chr(10).join(f"- {item}" for item in allowed_confidence)}

## How to Fill the Sheet

For each candidate image:

1. Open the image path shown in the CSV.
2. Read the model prediction and active learning reason.
3. Fill `human_decision`.
4. Fill `human_fine_label` only if the image fits an existing fine label.
5. Fill `human_coarse_label` if a broader label is suitable.
6. Fill `proposed_new_label` if the item represents a new local class.
7. Fill `human_confidence`.
8. Add a short explanation in `human_notes`.

## Decision Guide

| Case | human_decision |
|---|---|
| Image fits an existing fine label | known_label |
| Image is a new item type not covered by the taxonomy | new_unknown_class |
| Image contains multiple mixed objects/materials | mixed_waste |
| Image is too blurry, occluded, or unclear | unclear_image |
| Image should not be used | remove_sample |
"""

    output_path.write_text(text, encoding="utf-8")


def write_markdown_report(
    output_path: Path,
    summary: dict[str, Any],
    labelling_df: pd.DataFrame,
    instructions_filename: str,
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
        "candidate_rank",
        "sample_id",
        "object_description",
        "hierarchical_decision_type",
        "hierarchical_final_label",
        "active_learning_score",
        "human_decision",
        "human_fine_label",
        "human_coarse_label",
        "proposed_new_label",
    ]

    available_preview_columns = [
        column
        for column in preview_columns
        if column in labelling_df.columns
    ]

    preview_df = labelling_df[available_preview_columns].head(20).copy()

    report = f"""# Human Labelling Sheet v1 Report

## Purpose

This report documents the human labelling sheet created from the active learning candidates.

## Summary

{dataframe_to_markdown_table(summary_df)}

## Output Files

| File | Purpose |
|---|---|
| ml/outputs/metrics/human_labelling_sheet_v1.csv | CSV sheet for human annotation |
| ml/outputs/metrics/{instructions_filename} | labelling guide for the reviewer |

## Sheet Preview

{dataframe_to_markdown_table(preview_df)}

## Research Interpretation

This stage prepares the active learning candidates for human review.

The reviewed annotations can later be used to identify new local labels, confirm mixed or unclear waste cases, and decide which samples should be added to the next dataset version.
"""

    output_path.write_text(report, encoding="utf-8")


def run_human_labelling_sheet_builder(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    outputs = config["outputs"]
    annotation_config = config["annotation"]

    candidates_path = resolve_path(project_root, paths["active_learning_candidates_csv"])
    output_metrics_dir = resolve_path(project_root, paths["output_metrics_dir"])
    docs_results_dir = resolve_path(project_root, paths["docs_results_dir"])

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    docs_results_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_path.exists():
        raise FileNotFoundError(f"Active learning candidates file not found: {candidates_path}")

    candidates_df = pd.read_csv(candidates_path)

    labelling_df = build_human_labelling_sheet(candidates_df)
    validate_human_labelling_sheet(labelling_df)

    summary = build_labelling_summary(labelling_df)

    labelling_sheet_path = output_metrics_dir / outputs["human_labelling_sheet_csv"]
    instructions_path = output_metrics_dir / outputs["human_labelling_instructions_md"]
    report_path = docs_results_dir / outputs["markdown_report"]

    labelling_df.to_csv(labelling_sheet_path, index=False)

    write_human_labelling_instructions(
        output_path=instructions_path,
        annotation_config=annotation_config,
    )

    write_markdown_report(
        output_path=report_path,
        summary=summary,
        labelling_df=labelling_df,
        instructions_filename=outputs["human_labelling_instructions_md"],
    )

    print("Human labelling sheet created successfully.")
    print(f"Rows: {len(labelling_df)}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nCreated files:")
    print(f"- labelling sheet: {labelling_sheet_path}")
    print(f"- instructions: {instructions_path}")
    print(f"- thesis report: {report_path}")

    return {
        "summary": summary,
        "labelling_sheet": str(labelling_sheet_path),
        "instructions": str(instructions_path),
        "report": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build human labelling sheet from active learning candidates."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/human_labelling_sheet.yaml",
        help="Path to human labelling sheet YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    run_human_labelling_sheet_builder(
        config_path=args.config,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()