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


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def validate_required_sheet_columns(sheet_df: pd.DataFrame) -> bool:
    required_columns = {
        "candidate_rank",
        "sample_id",
        "image_path",
        "hierarchical_decision_type",
        "hierarchical_final_label",
        "active_learning_score",
        "human_decision",
        "human_fine_label",
        "human_coarse_label",
        "proposed_new_label",
        "human_confidence",
        "human_notes",
        "reviewed_by",
        "review_date",
    }

    missing_columns = required_columns - set(sheet_df.columns)

    if missing_columns:
        raise ValueError(
            f"Human labelling sheet is missing required columns: {sorted(missing_columns)}"
        )

    if sheet_df.empty:
        raise ValueError("Human labelling sheet cannot be empty.")

    if not sheet_df["sample_id"].is_unique:
        raise ValueError("sample_id values must be unique in the human labelling sheet.")

    return True


def validate_single_review_row(
    row: pd.Series,
    annotation_config: dict[str, Any],
) -> tuple[str, str]:
    """
    Validate one row.

    Returns:
        validation_status, validation_message
    """
    allowed_decisions = set(annotation_config["allowed_human_decisions"])
    allowed_fine_labels = set(annotation_config["allowed_fine_labels"])
    allowed_coarse_labels = set(annotation_config["allowed_coarse_labels"])
    allowed_confidence = set(annotation_config["allowed_human_confidence"])

    human_decision = clean_text(row.get("human_decision", ""))
    human_fine_label = clean_text(row.get("human_fine_label", ""))
    human_coarse_label = clean_text(row.get("human_coarse_label", ""))
    proposed_new_label = clean_text(row.get("proposed_new_label", ""))
    human_confidence = clean_text(row.get("human_confidence", ""))

    if not human_decision:
        return "pending_review", "No human decision entered yet."

    if human_decision not in allowed_decisions:
        return "invalid_review", f"Invalid human_decision: {human_decision}"

    if human_confidence and human_confidence not in allowed_confidence:
        return "invalid_review", f"Invalid human_confidence: {human_confidence}"

    if human_decision == "known_label":
        if not human_fine_label:
            return "invalid_review", "known_label requires human_fine_label."

        if not human_coarse_label:
            return "invalid_review", "known_label requires human_coarse_label."

        if human_fine_label not in allowed_fine_labels:
            return "invalid_review", f"Invalid human_fine_label: {human_fine_label}"

        if human_coarse_label not in allowed_coarse_labels:
            return "invalid_review", f"Invalid human_coarse_label: {human_coarse_label}"

        return "reviewed", "Known-label annotation is valid."

    if human_decision == "new_unknown_class":
        if not proposed_new_label:
            return "invalid_review", "new_unknown_class requires proposed_new_label."

        return "reviewed", "New local class annotation is valid."

    if human_decision in {"mixed_waste", "unclear_image", "remove_sample"}:
        return "reviewed", f"{human_decision} annotation is valid."

    return "invalid_review", "Unhandled review state."


def determine_dataset_action(
    review_status: str,
    human_decision: str,
) -> str:
    if review_status == "pending_review":
        return "pending_review"

    if review_status == "invalid_review":
        return "fix_annotation"

    if human_decision == "known_label":
        return "add_as_known_sample"

    if human_decision == "new_unknown_class":
        return "add_as_new_unknown_candidate"

    if human_decision == "mixed_waste":
        return "keep_as_mixed_review_sample"

    if human_decision == "unclear_image":
        return "keep_for_review_only"

    if human_decision == "remove_sample":
        return "remove_from_future_dataset"

    return "fix_annotation"


def process_reviewed_labels(
    sheet_df: pd.DataFrame,
    annotation_config: dict[str, Any],
) -> pd.DataFrame:
    validate_required_sheet_columns(sheet_df)

    result = sheet_df.copy()

    review_statuses: list[str] = []
    validation_messages: list[str] = []
    dataset_actions: list[str] = []
    ready_for_dataset_flags: list[bool] = []

    for _, row in result.iterrows():
        review_status, validation_message = validate_single_review_row(
            row=row,
            annotation_config=annotation_config,
        )

        human_decision = clean_text(row.get("human_decision", ""))
        dataset_action = determine_dataset_action(
            review_status=review_status,
            human_decision=human_decision,
        )

        ready_for_dataset = dataset_action in {
            "add_as_known_sample",
            "add_as_new_unknown_candidate",
        }

        review_statuses.append(review_status)
        validation_messages.append(validation_message)
        dataset_actions.append(dataset_action)
        ready_for_dataset_flags.append(bool(ready_for_dataset))

    result["review_status"] = review_statuses
    result["review_validation_message"] = validation_messages
    result["dataset_action"] = dataset_actions
    result["ready_for_dataset_v2"] = ready_for_dataset_flags

    return result


def build_ready_for_dataset_rows(processed_df: pd.DataFrame) -> pd.DataFrame:
    ready_df = processed_df[processed_df["ready_for_dataset_v2"] == True].copy()

    output_columns = [
        "sample_id",
        "image_path",
        "human_decision",
        "human_fine_label",
        "human_coarse_label",
        "proposed_new_label",
        "human_confidence",
        "human_notes",
        "reviewed_by",
        "review_date",
        "dataset_action",
    ]

    available_columns = [
        column
        for column in output_columns
        if column in ready_df.columns
    ]

    return ready_df[available_columns].copy()


def build_review_summary(processed_df: pd.DataFrame) -> dict[str, Any]:
    total_rows = int(len(processed_df))

    status_counts = processed_df["review_status"].value_counts().to_dict()
    action_counts = processed_df["dataset_action"].value_counts().to_dict()

    return {
        "total_rows": total_rows,
        "reviewed_rows": int(status_counts.get("reviewed", 0)),
        "pending_review_rows": int(status_counts.get("pending_review", 0)),
        "invalid_review_rows": int(status_counts.get("invalid_review", 0)),
        "ready_for_dataset_rows": int(processed_df["ready_for_dataset_v2"].sum()),
        "add_as_known_sample_rows": int(action_counts.get("add_as_known_sample", 0)),
        "add_as_new_unknown_candidate_rows": int(
            action_counts.get("add_as_new_unknown_candidate", 0)
        ),
        "keep_as_mixed_review_sample_rows": int(
            action_counts.get("keep_as_mixed_review_sample", 0)
        ),
        "keep_for_review_only_rows": int(action_counts.get("keep_for_review_only", 0)),
        "remove_from_future_dataset_rows": int(
            action_counts.get("remove_from_future_dataset", 0)
        ),
        "fix_annotation_rows": int(action_counts.get("fix_annotation", 0)),
    }


def write_markdown_report(
    output_path: Path,
    summary: dict[str, Any],
    processed_df: pd.DataFrame,
    ready_df: pd.DataFrame,
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

    status_df = (
        processed_df["review_status"]
        .value_counts()
        .rename_axis("review_status")
        .reset_index(name="count")
    )

    action_df = (
        processed_df["dataset_action"]
        .value_counts()
        .rename_axis("dataset_action")
        .reset_index(name="count")
    )

    preview_columns = [
        "candidate_rank",
        "sample_id",
        "human_decision",
        "human_fine_label",
        "human_coarse_label",
        "proposed_new_label",
        "review_status",
        "dataset_action",
        "review_validation_message",
    ]

    available_preview_columns = [
        column
        for column in preview_columns
        if column in processed_df.columns
    ]

    preview_df = processed_df[available_preview_columns].head(20).copy()

    report = f"""# Reviewed Human Label Processing v1 Report

## Purpose

This report processes the human labelling sheet and converts it into review statuses and dataset actions.

## Summary

{dataframe_to_markdown_table(summary_df)}

## Review Status Counts

{dataframe_to_markdown_table(status_df)}

## Dataset Action Counts

{dataframe_to_markdown_table(action_df)}

## Processed Sheet Preview

{dataframe_to_markdown_table(preview_df)}

## Ready for Dataset Rows

Ready rows: {len(ready_df)}

## Research Interpretation

This stage validates and processes human annotations from the active learning workflow.

If the labelling sheet has not yet been filled, the expected result is that all rows remain pending review. After human annotation, this same script can be rerun to identify samples that are ready to be added to the next dataset version.
"""

    output_path.write_text(report, encoding="utf-8")


def run_reviewed_label_processing(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    outputs = config["outputs"]
    annotation_config = config["annotation"]

    sheet_path = resolve_path(project_root, paths["human_labelling_sheet_csv"])
    output_metrics_dir = resolve_path(project_root, paths["output_metrics_dir"])
    docs_results_dir = resolve_path(project_root, paths["docs_results_dir"])

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    docs_results_dir.mkdir(parents=True, exist_ok=True)

    if not sheet_path.exists():
        raise FileNotFoundError(f"Human labelling sheet not found: {sheet_path}")

    sheet_df = pd.read_csv(sheet_path)

    processed_df = process_reviewed_labels(
        sheet_df=sheet_df,
        annotation_config=annotation_config,
    )

    ready_df = build_ready_for_dataset_rows(processed_df)
    summary = build_review_summary(processed_df)

    reviewed_decisions_path = output_metrics_dir / outputs["reviewed_decisions_csv"]
    ready_for_dataset_path = output_metrics_dir / outputs["ready_for_dataset_csv"]
    summary_path = output_metrics_dir / outputs["summary_json"]
    report_path = docs_results_dir / outputs["markdown_report"]

    processed_df.to_csv(reviewed_decisions_path, index=False)
    ready_df.to_csv(ready_for_dataset_path, index=False)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        output_path=report_path,
        summary=summary,
        processed_df=processed_df,
        ready_df=ready_df,
    )

    print("Reviewed human label processing completed successfully.")
    print(f"Rows processed: {len(processed_df)}")
    print(f"Ready for dataset rows: {len(ready_df)}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nCreated files:")
    print(f"- reviewed decisions: {reviewed_decisions_path}")
    print(f"- ready for dataset: {ready_for_dataset_path}")
    print(f"- summary: {summary_path}")
    print(f"- thesis report: {report_path}")

    return {
        "summary": summary,
        "reviewed_decisions": str(reviewed_decisions_path),
        "ready_for_dataset": str(ready_for_dataset_path),
        "report": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process reviewed human labels from active learning sheet."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/reviewed_label_processing.yaml",
        help="Path to reviewed label processing YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    run_reviewed_label_processing(
        config_path=args.config,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()