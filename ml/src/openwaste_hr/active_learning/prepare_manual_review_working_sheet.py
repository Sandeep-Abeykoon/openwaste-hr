from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IGNORED_REVIEW_VALUES = {
    "",
    "none",
    "nan",
    "pending",
    "todo",
    "tbd",
    "to_review",
    "not_reviewed",
    "n/a",
}


OUTPUT_FIELDS = [
    "source_file",
    "sample_id",
    "image_path",
    "current_model_label",
    "current_model_confidence",
    "current_model_decision_type",
    "selection_reason",
    "human_observation",
    "taxonomy_status",
    "reviewed_fine_label",
    "reviewed_coarse_label",
    "recommended_action",
    "active_learning_v2_role",
    "reviewer_notes",
    "review_status",
]


PREFERRED_CANDIDATE_FILENAMES = [
    "human_labelling_sheet_v1_seeded_review.csv",
    "human_labelling_sheet_v1_working_review.csv",
    "human_labelling_sheet_v1.csv",
    "active_learning_candidates_v1.csv",
]


BLOCKED_FILENAME_KEYWORDS = [
    "audit",
    "working_sheet",
    "distribution",
    "dataset_plan",
    "ready_for_dataset",
    "decisions",
    "predictions",
    "manifest",
    "unknown_test",
    "accepted_label_distribution",
]


@dataclass(frozen=True)
class WorkingSheetSummary:
    total_rows: int
    pending_review_rows: int
    already_reviewed_rows: int


def normalise_key(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def normalise_row(row: dict[str, str]) -> dict[str, str]:
    return {normalise_key(key): str(value).strip() for key, value in row.items()}


def first_non_empty(row: dict[str, str], candidate_keys: Iterable[str]) -> str:
    normalised = normalise_row(row)

    for key in candidate_keys:
        value = normalised.get(normalise_key(key), "").strip()
        if value:
            return value

    return ""


def has_review_signal(row: dict[str, str]) -> bool:
    review_values = [
        first_non_empty(row, ["human_observation", "human_description", "reviewer_observation"]),
        first_non_empty(row, ["taxonomy_status"]),
        first_non_empty(row, ["reviewed_fine_label", "human_fine_label"]),
        first_non_empty(row, ["reviewed_coarse_label", "human_coarse_label"]),
        first_non_empty(row, ["recommended_action", "action"]),
        first_non_empty(row, ["active_learning_v2_role", "active_learning_role", "role"]),
        first_non_empty(row, ["reviewer_notes", "notes"]),
    ]

    return any(value.strip().lower() not in IGNORED_REVIEW_VALUES for value in review_values)


def infer_review_status(row: dict[str, str]) -> str:
    if has_review_signal(row):
        return "already_reviewed"

    return "pending_review"


def discover_all_csv_files(project_root: Path) -> list[Path]:
    search_roots = [
        project_root / "ml" / "outputs" / "metrics",
        project_root / "ml" / "data" / "splits",
        project_root / "ml" / "outputs" / "active_learning",
    ]

    csv_files: list[Path] = []

    for search_root in search_roots:
        if not search_root.exists():
            continue

        csv_files.extend(search_root.rglob("*.csv"))

    return sorted(set(csv_files))


def discover_candidate_files(project_root: Path) -> list[Path]:
    csv_files = discover_all_csv_files(project_root)

    for preferred_name in PREFERRED_CANDIDATE_FILENAMES:
        matches = [
            csv_path
            for csv_path in csv_files
            if csv_path.name.lower() == preferred_name
        ]

        if matches:
            return sorted(matches)

    fallback_files: list[Path] = []

    for csv_path in csv_files:
        name = csv_path.name.lower()

        if any(blocked in name for blocked in BLOCKED_FILENAME_KEYWORDS):
            continue

        if any(
            keyword in name
            for keyword in [
                "candidate",
                "human",
                "review",
                "labelling",
                "labeling",
                "active_learning",
            ]
        ):
            fallback_files.append(csv_path)

    return sorted(set(fallback_files))


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            return []

        return list(reader)


def make_working_sheet_row(
    project_root: Path,
    source_file: Path,
    row: dict[str, str],
) -> dict[str, str]:
    source_relative = str(source_file.relative_to(project_root))

    return {
        "source_file": source_relative,
        "sample_id": first_non_empty(row, ["sample_id", "id"]),
        "image_path": first_non_empty(row, ["image_path", "path"]),
        "current_model_label": first_non_empty(
            row,
            [
                "current_model_label",
                "model_predicted_label",
                "predicted_label",
                "predicted_fine_label",
                "final_label",
                "top_label",
            ],
        ),
        "current_model_confidence": first_non_empty(
            row,
            [
                "current_model_confidence",
                "model_confidence",
                "confidence",
                "max_softmax_confidence",
                "final_confidence",
                "score",
            ],
        ),
        "current_model_decision_type": first_non_empty(
            row,
            [
                "current_model_decision_type",
                "model_decision_type",
                "decision_type",
                "final_decision_type",
            ],
        ),
        "selection_reason": first_non_empty(
            row,
            [
                "selection_reason",
                "candidate_reason",
                "uncertainty_reason",
                "active_learning_reason",
                "reason",
            ],
        ),
        "human_observation": first_non_empty(
            row,
            ["human_observation", "human_description", "reviewer_observation"],
        ),
        "taxonomy_status": first_non_empty(row, ["taxonomy_status"]),
        "reviewed_fine_label": first_non_empty(
            row,
            ["reviewed_fine_label", "human_fine_label"],
        ),
        "reviewed_coarse_label": first_non_empty(
            row,
            ["reviewed_coarse_label", "human_coarse_label"],
        ),
        "recommended_action": first_non_empty(row, ["recommended_action", "action"]),
        "active_learning_v2_role": first_non_empty(
            row,
            ["active_learning_v2_role", "active_learning_role", "role"],
        ),
        "reviewer_notes": first_non_empty(row, ["reviewer_notes", "notes"]),
        "review_status": infer_review_status(row),
    }


def build_working_sheet_rows(project_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    for source_file in discover_candidate_files(project_root):
        for row in read_csv_rows(source_file):
            sheet_row = make_working_sheet_row(project_root, source_file, row)

            dedupe_key = (
                sheet_row["sample_id"],
                sheet_row["image_path"],
            )

            if dedupe_key in seen_keys:
                continue

            seen_keys.add(dedupe_key)
            rows.append(sheet_row)

    return rows


def write_working_sheet(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def summarise_rows(rows: list[dict[str, str]]) -> WorkingSheetSummary:
    pending_review_rows = sum(
        1 for row in rows if row.get("review_status") == "pending_review"
    )
    already_reviewed_rows = sum(
        1 for row in rows if row.get("review_status") == "already_reviewed"
    )

    return WorkingSheetSummary(
        total_rows=len(rows),
        pending_review_rows=pending_review_rows,
        already_reviewed_rows=already_reviewed_rows,
    )


def write_markdown_report(rows: list[dict[str, str]], output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    summary = summarise_rows(rows)

    content = f"""# Manual Review Working Sheet v1

## Purpose

This report prepares the manual review worksheet for OpenWaste-HR active learning v2.

The working sheet collects candidate images that need human review before active learning retraining can be considered.

## Summary

| Metric | Value |
|---|---:|
| total working sheet rows | {summary.total_rows} |
| pending review rows | {summary.pending_review_rows} |
| already reviewed rows | {summary.already_reviewed_rows} |

## Output CSV

The working sheet was written to:

ml/outputs/active_learning/manual_review_working_sheet_v1.csv

## Manual Review Fields to Complete

For each pending row, complete these fields:

| Field | Allowed / Expected Value |
|---|---|
| human_observation | short description of the object |
| taxonomy_status | current_known_taxonomy, outside_current_known_taxonomy, or unclear |
| reviewed_fine_label | paper_cardboard, plastic, glass, metal, organic, residual, or blank if unknown |
| reviewed_coarse_label | recyclable, organic, residual, or blank if unknown |
| recommended_action | known_train_candidate, known_eval_candidate, unknown_test_candidate, future_class_candidate, or exclude_or_review_again |
| active_learning_v2_role | known_retraining_candidate, known_evaluation_candidate, unknown_test_and_future_class_candidate, or excluded |
| reviewer_notes | short justification |

## Important Rule

Do not force outside-taxonomy samples into known classes.

Only samples that clearly belong to paper_cardboard, plastic, glass, metal, organic, or residual should be used for known-class retraining.

## Next Step

After the worksheet is completed, rerun the manual review audit.

If enough valid known-class samples exist, active learning retraining can proceed.

If not, the reviewed records should be reported as workflow evidence and unknown/future-class analysis.
"""

    output_md.write_text(content, encoding="utf-8")


def write_supervisor_summary(rows: list[dict[str, str]], output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    summary = summarise_rows(rows)

    content = f"""# Manual Review Working Sheet Summary v1

## Purpose

This stage prepared a manual review working sheet for OpenWaste-HR active learning v2.

## Result

| Metric | Value |
|---|---:|
| total working sheet rows | {summary.total_rows} |
| pending review rows | {summary.pending_review_rows} |
| already reviewed rows | {summary.already_reviewed_rows} |

## Meaning

The working sheet allows local candidate images to be reviewed safely before retraining.

The key purpose is to separate valid known-class retraining samples from unknown or future-class samples.

## Active Learning Impact

The impact of active learning can only be measured after enough reviewed known-class samples are available.

Until then, the project should report active learning as a validated human-review workflow and dataset-quality protection mechanism.
"""

    output_md.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a manual review working sheet for active learning v2."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ml/outputs/active_learning/manual_review_working_sheet_v1.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/results/manual_review_working_sheet_v1.md"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("docs/results/manual_review_working_sheet_summary_v1.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    rows = build_working_sheet_rows(project_root)

    write_working_sheet(rows, project_root / args.output_csv)
    write_markdown_report(rows, project_root / args.output_md)
    write_supervisor_summary(rows, project_root / args.summary_md)

    summary = summarise_rows(rows)

    print("Manual review working sheet prepared")
    print(f"Total working sheet rows: {summary.total_rows}")
    print(f"Pending review rows: {summary.pending_review_rows}")
    print(f"Already reviewed rows: {summary.already_reviewed_rows}")


if __name__ == "__main__":
    main()
