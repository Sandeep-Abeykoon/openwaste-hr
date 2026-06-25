from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


KNOWN_CLASSES = {"cardboard", "glass", "metal", "paper", "plastic"}


def normalize_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def normalize_action(value: str) -> str:
    return value.strip().lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_known_addition_row(row: dict[str, str], stage_name: str) -> dict[str, str]:
    reviewed_label = normalize_label(row["reviewed_label"])

    return {
        "source_dataset": row["source_dataset"],
        "image_path": row["image_path"],
        "source_label": row["source_label"],
        "canonical_label": reviewed_label,
        "split_role": "known",
        "include_in_known_training": "true",
        "notes": (
            f"{stage_name} active-learning human-reviewed known addition. "
            f"Original source reference label: {row['source_reference_label']}. "
            f"Model prediction: {row['model_prediction']}."
        ),
        "split": "known_train",
        "active_learning_stage": stage_name,
        "review_id": row["review_id"],
        "human_observation": row["human_observation"],
        "reviewed_label": reviewed_label,
        "reviewed_action": row["reviewed_action"],
        "reviewer_notes": row["reviewer_notes"],
    }


def build_excluded_row(row: dict[str, str], reason: str) -> dict[str, str]:
    return {
        "review_id": row.get("review_id", ""),
        "image_path": row.get("image_path", ""),
        "source_dataset": row.get("source_dataset", ""),
        "source_reference_label": row.get("source_reference_label", ""),
        "model_prediction": row.get("model_prediction", ""),
        "model_confidence": row.get("model_confidence", ""),
        "selection_reason": row.get("selection_reason", ""),
        "human_observation": row.get("human_observation", ""),
        "reviewed_label": row.get("reviewed_label", ""),
        "reviewed_action": row.get("reviewed_action", ""),
        "review_status": row.get("review_status", ""),
        "exclusion_reason": reason,
        "reviewer_notes": row.get("reviewer_notes", ""),
    }


def process_review_sheet(
    *,
    input_csv: Path,
    known_output_csv: Path,
    excluded_output_csv: Path,
    summary_json: Path,
    stage_name: str,
) -> None:
    rows = read_csv(input_csv)

    known_additions: list[dict[str, str]] = []
    excluded_or_unknown: list[dict[str, str]] = []
    invalid_or_pending: list[dict[str, str]] = []

    for row in rows:
        review_status = normalize_action(row.get("review_status", ""))
        reviewed_action = normalize_action(row.get("reviewed_action", ""))
        reviewed_label = normalize_label(row.get("reviewed_label", ""))

        if review_status != "reviewed":
            invalid_or_pending.append(build_excluded_row(row, "not_reviewed_or_pending"))
            continue

        if reviewed_action == "add_to_training":
            if reviewed_label not in KNOWN_CLASSES:
                invalid_or_pending.append(
                    build_excluded_row(row, "add_to_training_but_label_not_known_class")
                )
                continue

            known_additions.append(build_known_addition_row(row, stage_name))
            continue

        if reviewed_action in {"exclude_or_unknown", "exclude", "unknown", "manual_review_only"}:
            excluded_or_unknown.append(build_excluded_row(row, "reviewed_as_exclude_or_unknown"))
            continue

        invalid_or_pending.append(build_excluded_row(row, "invalid_or_missing_reviewed_action"))

    all_excluded = excluded_or_unknown + invalid_or_pending

    known_fieldnames = [
        "source_dataset",
        "image_path",
        "source_label",
        "canonical_label",
        "split_role",
        "include_in_known_training",
        "notes",
        "split",
        "active_learning_stage",
        "review_id",
        "human_observation",
        "reviewed_label",
        "reviewed_action",
        "reviewer_notes",
    ]

    excluded_fieldnames = [
        "review_id",
        "image_path",
        "source_dataset",
        "source_reference_label",
        "model_prediction",
        "model_confidence",
        "selection_reason",
        "human_observation",
        "reviewed_label",
        "reviewed_action",
        "review_status",
        "exclusion_reason",
        "reviewer_notes",
    ]

    write_csv(known_output_csv, known_additions, known_fieldnames)
    write_csv(excluded_output_csv, all_excluded, excluded_fieldnames)

    summary = {
        "stage_name": stage_name,
        "input_review_sheet": str(input_csv),
        "total_review_rows": len(rows),
        "known_additions": len(known_additions),
        "excluded_or_unknown": len(excluded_or_unknown),
        "invalid_or_pending": len(invalid_or_pending),
        "known_additions_by_label": dict(
            sorted(Counter(row["canonical_label"] for row in known_additions).items())
        ),
        "excluded_reason_counts": dict(
            sorted(Counter(row["exclusion_reason"] for row in all_excluded).items())
        ),
        "rule": (
            "Only rows marked review_status=reviewed, reviewed_action=add_to_training, "
            "and reviewed_label in cardboard/glass/metal/paper/plastic are added to training."
        ),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Known additions written: {known_output_csv}")
    print(f"Excluded/unknown written: {excluded_output_csv}")
    print(f"Summary written: {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process active-learning human review sheet."
    )

    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--known-output-csv", type=Path, required=True)
    parser.add_argument("--excluded-output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--stage-name", type=str, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    process_review_sheet(
        input_csv=args.input_csv,
        known_output_csv=args.known_output_csv,
        excluded_output_csv=args.excluded_output_csv,
        summary_json=args.summary_json,
        stage_name=args.stage_name,
    )


if __name__ == "__main__":
    main()
