from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


CONFIRMED_SAMPLE_ID = "local_000001"
HUMAN_OBSERVED_OBJECT = "rubber slipper / flip-flop"
HUMAN_TAXONOMY_STATUS = "outside_current_known_taxonomy"
RECOMMENDED_ACTION = "keep_as_unknown_test"
PROPOSED_NEW_LABEL = "rubber_slipper_flip_flop"
REVIEWER_NAME = "Jason Bernard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reviewed local-label seed outputs for OpenWaste-HR."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory. Default: current directory.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    for column in required_columns:
        if column not in df.columns:
            df[column] = ""
    return df


def normalise_empty_cells(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna("")


def write_markdown_report(
    report_path: Path,
    *,
    summary: dict[str, Any],
    reviewed_record: dict[str, Any],
) -> None:
    report = f"""# Reviewed Local Label Seed v1 Report

## Purpose

This report records the first confirmed human-reviewed local unknown sample for OpenWaste-HR active learning v2.

## Summary

| metric | value |
| --- | --- |
| total_sheet_rows | {summary["total_sheet_rows"]} |
| seeded_review_rows | {summary["seeded_review_rows"]} |
| pending_review_rows | {summary["pending_review_rows"]} |
| ready_for_active_learning_v2_rows | {summary["ready_for_active_learning_v2_rows"]} |

## Reviewed Seed Record

| field | value |
| --- | --- |
| sample_id | {reviewed_record["sample_id"]} |
| image_path | {reviewed_record["image_path"]} |
| human_observed_object | {reviewed_record["human_observed_object"]} |
| human_taxonomy_status | {reviewed_record["human_taxonomy_status"]} |
| recommended_action | {reviewed_record["recommended_action"]} |
| proposed_new_label | {reviewed_record["proposed_new_label"]} |
| active_learning_v2_role | {reviewed_record["active_learning_v2_role"]} |

## Research Interpretation

The reviewed seed confirms that local_000001 is a rubber slipper / flip-flop, which is outside the current known fine-label taxonomy.

This sample should not be forced into paper_cardboard, plastic, glass, metal, organic, e_waste_battery, or residual. It should be kept as an unknown-test sample and future-class candidate for later active-learning work.

This supports the OpenWaste-HR motivation: real-world waste classification requires local unknown evaluation, manual-review routing, and human-in-the-loop active learning.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    input_sheet_path = (
        project_root / "ml" / "outputs" / "metrics" / "human_labelling_sheet_v1_working_review.csv"
    )

    if not input_sheet_path.exists():
        raise FileNotFoundError(
            f"Working review sheet not found: {input_sheet_path}. "
            "Create it from human_labelling_sheet_v1.csv first."
        )

    output_metrics_dir = project_root / "ml" / "outputs" / "metrics"
    output_data_dir = project_root / "ml" / "data" / "splits"
    output_report_dir = project_root / "docs" / "results"

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    seeded_review_sheet_path = (
        output_metrics_dir / "human_labelling_sheet_v1_seeded_review.csv"
    )
    reviewed_seed_metrics_path = output_metrics_dir / "reviewed_local_labels_seed_v1.csv"
    reviewed_seed_data_path = output_data_dir / "reviewed_local_labels_seed_v1.csv"
    summary_path = output_metrics_dir / "reviewed_local_labels_seed_summary_v1.json"
    report_path = output_report_dir / "reviewed_local_labels_seed_v1_report.md"

    df = pd.read_csv(input_sheet_path)
    df = normalise_empty_cells(df)

    required_columns = [
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
        "recommended_human_action",
    ]
    df = ensure_columns(df, required_columns)

    if CONFIRMED_SAMPLE_ID not in set(df["sample_id"].astype(str)):
        raise ValueError(
            f"{CONFIRMED_SAMPLE_ID} was not found in {input_sheet_path}. "
            "Check the active-learning sheet before continuing."
        )

    today = date.today().isoformat()
    sample_mask = df["sample_id"].astype(str) == CONFIRMED_SAMPLE_ID

    df.loc[sample_mask, "human_decision"] = HUMAN_TAXONOMY_STATUS
    df.loc[sample_mask, "human_fine_label"] = ""
    df.loc[sample_mask, "human_coarse_label"] = ""
    df.loc[sample_mask, "proposed_new_label"] = PROPOSED_NEW_LABEL
    df.loc[sample_mask, "human_confidence"] = 1.0
    df.loc[sample_mask, "human_notes"] = (
        "Human reviewed: rubber slipper / flip-flop. "
        "Outside current known taxonomy; keep as unknown-test sample."
    )
    df.loc[sample_mask, "reviewed_by"] = REVIEWER_NAME
    df.loc[sample_mask, "review_date"] = today
    df.loc[sample_mask, "recommended_human_action"] = RECOMMENDED_ACTION

    df.to_csv(seeded_review_sheet_path, index=False)

    reviewed_row = df.loc[sample_mask].iloc[0].to_dict()

    reviewed_record = {
        "sample_id": CONFIRMED_SAMPLE_ID,
        "image_path": reviewed_row.get(
            "image_path", "ml/data/local_unknown/images/local_000001.jpg"
        ),
        "human_observed_object": HUMAN_OBSERVED_OBJECT,
        "human_taxonomy_status": HUMAN_TAXONOMY_STATUS,
        "recommended_action": RECOMMENDED_ACTION,
        "proposed_new_label": PROPOSED_NEW_LABEL,
        "human_confidence": 1.0,
        "reviewed_by": REVIEWER_NAME,
        "review_date": today,
        "active_learning_v2_role": "unknown_test_and_future_class_candidate",
        "is_ready_for_active_learning_v2": True,
        "notes": (
            "Confirmed human observation. The object is not part of the current "
            "known fine-label taxonomy."
        ),
    }

    reviewed_seed_df = pd.DataFrame([reviewed_record])
    reviewed_seed_df.to_csv(reviewed_seed_metrics_path, index=False)
    reviewed_seed_df.to_csv(reviewed_seed_data_path, index=False)

    seeded_review_rows = 1
    total_sheet_rows = int(len(df))
    pending_review_rows = int(total_sheet_rows - seeded_review_rows)

    summary = {
        "total_sheet_rows": total_sheet_rows,
        "seeded_review_rows": seeded_review_rows,
        "pending_review_rows": pending_review_rows,
        "ready_for_active_learning_v2_rows": 1,
        "confirmed_sample_id": CONFIRMED_SAMPLE_ID,
        "human_observed_object": HUMAN_OBSERVED_OBJECT,
        "human_taxonomy_status": HUMAN_TAXONOMY_STATUS,
        "recommended_action": RECOMMENDED_ACTION,
        "proposed_new_label": PROPOSED_NEW_LABEL,
        "outputs": {
            "seeded_review_sheet": str(seeded_review_sheet_path.relative_to(project_root)),
            "reviewed_seed_metrics_csv": str(reviewed_seed_metrics_path.relative_to(project_root)),
            "reviewed_seed_data_csv": str(reviewed_seed_data_path.relative_to(project_root)),
            "summary_json": str(summary_path.relative_to(project_root)),
            "markdown_report": str(report_path.relative_to(project_root)),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_markdown_report(
        report_path,
        summary=summary,
        reviewed_record=reviewed_record,
    )

    print("Reviewed local label seed created successfully.")
    print(f"Total sheet rows: {total_sheet_rows}")
    print(f"Seeded reviewed rows: {seeded_review_rows}")
    print(f"Pending review rows: {pending_review_rows}")
    print()
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print()
    print("Created files:")
    print(f"- seeded review sheet: {seeded_review_sheet_path.relative_to(project_root)}")
    print(f"- reviewed seed metrics CSV: {reviewed_seed_metrics_path.relative_to(project_root)}")
    print(f"- reviewed seed data CSV: {reviewed_seed_data_path.relative_to(project_root)}")
    print(f"- summary JSON: {summary_path.relative_to(project_root)}")
    print(f"- thesis report: {report_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()