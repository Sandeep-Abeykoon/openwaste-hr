from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


KNOWN_TAXONOMY_STATUS = "inside_current_known_taxonomy"
UNKNOWN_TAXONOMY_STATUS = "outside_current_known_taxonomy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build active learning v2 dataset plan from reviewed local labels."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory. Default: current directory.",
    )
    return parser.parse_args()


def decide_dataset_role(row: pd.Series) -> dict[str, Any]:
    taxonomy_status = str(row.get("human_taxonomy_status", "")).strip()
    recommended_action = str(row.get("recommended_action", "")).strip()
    proposed_new_label = str(row.get("proposed_new_label", "")).strip()

    include_in_known_training = False
    include_in_unknown_test = False
    include_as_future_class_candidate = False
    exclude_from_dataset_v2 = False
    recollection_candidate = False
    active_learning_v2_role = "manual_review_pending"

    if taxonomy_status == KNOWN_TAXONOMY_STATUS:
        include_in_known_training = True
        active_learning_v2_role = "add_to_known_training_candidate"

    elif taxonomy_status == UNKNOWN_TAXONOMY_STATUS:
        include_in_unknown_test = True
        include_as_future_class_candidate = bool(proposed_new_label)
        active_learning_v2_role = "unknown_test_and_future_class_candidate"

    if recommended_action == "recollect_image":
        include_in_known_training = False
        include_in_unknown_test = False
        include_as_future_class_candidate = False
        recollection_candidate = True
        active_learning_v2_role = "recollection_candidate"

    if recommended_action == "exclude_duplicate":
        include_in_known_training = False
        include_in_unknown_test = False
        include_as_future_class_candidate = False
        exclude_from_dataset_v2 = True
        active_learning_v2_role = "exclude_from_dataset_v2"

    return {
        "include_in_known_training_v2": include_in_known_training,
        "include_in_unknown_test_v2": include_in_unknown_test,
        "include_as_future_class_candidate": include_as_future_class_candidate,
        "recollection_candidate": recollection_candidate,
        "exclude_from_dataset_v2": exclude_from_dataset_v2,
        "active_learning_v2_role": active_learning_v2_role,
    }


def write_markdown_report(
    report_path: Path,
    *,
    summary: dict[str, Any],
    plan_df: pd.DataFrame,
) -> None:
    role_counts = plan_df["active_learning_v2_role"].value_counts().to_dict()

    role_rows = "\n".join(
        f"| {role} | {count} |" for role, count in sorted(role_counts.items())
    )

    sample_rows = "\n".join(
        (
            f"| {row.sample_id} | {row.human_observed_object} | "
            f"{row.human_taxonomy_status} | {row.active_learning_v2_role} |"
        )
        for row in plan_df.itertuples(index=False)
    )

    report = f"""# Active Learning v2 Dataset Plan v1 Report

## Purpose

This report records the first active learning v2 dataset plan for OpenWaste-HR.

## Summary

| metric | value |
| --- | ---: |
| total_reviewed_seed_rows | {summary["total_reviewed_seed_rows"]} |
| known_training_candidates | {summary["known_training_candidates"]} |
| unknown_test_candidates | {summary["unknown_test_candidates"]} |
| future_class_candidates | {summary["future_class_candidates"]} |
| recollection_candidates | {summary["recollection_candidates"]} |
| excluded_candidates | {summary["excluded_candidates"]} |

## Active Learning v2 Role Counts

| role | count |
| --- | ---: |
{role_rows}

## Planned Samples

| sample_id | human_observed_object | human_taxonomy_status | active_learning_v2_role |
| --- | --- | --- | --- |
{sample_rows}

## Research Interpretation

The reviewed local seed is treated as an unknown-test and future-class candidate rather than being forced into an existing known class.

This supports the OpenWaste-HR aim of safer open-set waste classification with human-in-the-loop active learning.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    input_path = project_root / "ml" / "data" / "splits" / "reviewed_local_labels_seed_v1.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Reviewed local label seed not found: {input_path}. "
            "Run Step 42 before building the active learning v2 dataset plan."
        )

    output_data_dir = project_root / "ml" / "data" / "splits"
    output_metrics_dir = project_root / "ml" / "outputs" / "metrics"
    output_report_dir = project_root / "docs" / "results"

    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    plan_data_path = output_data_dir / "active_learning_v2_dataset_plan_v1.csv"
    plan_metrics_path = output_metrics_dir / "active_learning_v2_dataset_plan_v1.csv"
    summary_path = output_metrics_dir / "active_learning_v2_dataset_plan_summary_v1.json"
    report_path = output_report_dir / "active_learning_v2_dataset_plan_v1_report.md"

    reviewed_df = pd.read_csv(input_path).fillna("")

    plan_records: list[dict[str, Any]] = []

    for _, row in reviewed_df.iterrows():
        role_decision = decide_dataset_role(row)

        plan_records.append(
            {
                "sample_id": row.get("sample_id", ""),
                "image_path": row.get("image_path", ""),
                "human_observed_object": row.get("human_observed_object", ""),
                "human_taxonomy_status": row.get("human_taxonomy_status", ""),
                "recommended_action": row.get("recommended_action", ""),
                "proposed_new_label": row.get("proposed_new_label", ""),
                "human_confidence": row.get("human_confidence", ""),
                "reviewed_by": row.get("reviewed_by", ""),
                "review_date": row.get("review_date", ""),
                **role_decision,
            }
        )

    plan_df = pd.DataFrame(plan_records)

    plan_df.to_csv(plan_data_path, index=False)
    plan_df.to_csv(plan_metrics_path, index=False)

    summary = {
        "total_reviewed_seed_rows": int(len(plan_df)),
        "known_training_candidates": int(plan_df["include_in_known_training_v2"].sum()),
        "unknown_test_candidates": int(plan_df["include_in_unknown_test_v2"].sum()),
        "future_class_candidates": int(
            plan_df["include_as_future_class_candidate"].sum()
        ),
        "recollection_candidates": int(plan_df["recollection_candidate"].sum()),
        "excluded_candidates": int(plan_df["exclude_from_dataset_v2"].sum()),
        "active_learning_v2_roles": {
            str(role): int(count)
            for role, count in plan_df["active_learning_v2_role"].value_counts().items()
        },
        "outputs": {
            "plan_data_csv": str(plan_data_path.relative_to(project_root)),
            "plan_metrics_csv": str(plan_metrics_path.relative_to(project_root)),
            "summary_json": str(summary_path.relative_to(project_root)),
            "markdown_report": str(report_path.relative_to(project_root)),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_markdown_report(report_path, summary=summary, plan_df=plan_df)

    print("Active learning v2 dataset plan created successfully.")
    print(f"Total reviewed seed rows: {summary['total_reviewed_seed_rows']}")
    print(f"Known training candidates: {summary['known_training_candidates']}")
    print(f"Unknown test candidates: {summary['unknown_test_candidates']}")
    print(f"Future class candidates: {summary['future_class_candidates']}")
    print()
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print()
    print("Created files:")
    print(f"- plan data CSV: {plan_data_path.relative_to(project_root)}")
    print(f"- plan metrics CSV: {plan_metrics_path.relative_to(project_root)}")
    print(f"- summary JSON: {summary_path.relative_to(project_root)}")
    print(f"- thesis report: {report_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()