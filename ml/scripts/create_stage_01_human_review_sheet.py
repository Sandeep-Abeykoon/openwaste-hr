from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PRIORITY_ORDER = {
    "confident_wrong_candidate": 0,
    "low_confidence_candidate": 1,
    "low_margin_candidate": 2,
    "not_priority": 3,
}


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


def candidate_sort_key(row: dict[str, str]) -> tuple[int, float, float, int]:
    selection_reason = row.get("selection_reason", "not_priority")
    confidence = float(row.get("confidence", 1.0))
    probability_margin = float(row.get("probability_margin", 1.0))
    rank = int(row.get("active_learning_rank", "999999"))

    return (
        PRIORITY_ORDER.get(selection_reason, 99),
        confidence,
        probability_margin,
        rank,
    )


def select_balanced_candidates(
    rows: list[dict[str, str]],
    total_limit: int,
    per_label_minimum_target: int,
) -> list[dict[str, str]]:
    priority_rows = [
        row for row in rows
        if row.get("selection_reason") != "not_priority"
    ]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in priority_rows:
        grouped[row["true_label"]].append(row)

    for label in grouped:
        grouped[label].sort(key=candidate_sort_key)

    selected: list[dict[str, str]] = []
    selected_paths: set[str] = set()

    for label in sorted(grouped):
        label_rows = grouped[label]
        for row in label_rows[:per_label_minimum_target]:
            image_path = row["image_path"]
            if image_path not in selected_paths:
                selected.append(row)
                selected_paths.add(image_path)

    remaining_rows = sorted(priority_rows, key=candidate_sort_key)

    for row in remaining_rows:
        if len(selected) >= total_limit:
            break

        image_path = row["image_path"]

        if image_path in selected_paths:
            continue

        selected.append(row)
        selected_paths.add(image_path)

    return selected[:total_limit]


def build_review_rows(selected_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    review_rows: list[dict[str, str]] = []

    for index, row in enumerate(selected_rows, start=1):
        review_rows.append(
            {
                "review_id": f"stage01_al_{index:04d}",
                "image_path": row["image_path"],
                "source_dataset": row["source_dataset"],
                "source_label": row["source_label"],
                "source_reference_label": row["true_label"],
                "model_prediction": row["pred_label"],
                "model_confidence": row["confidence"],
                "model_max_logit": row["max_logit"],
                "model_energy": row["energy"],
                "probability_margin": row["probability_margin"],
                "prediction_matches_source_label": row["prediction_matches_source_label"],
                "uncertainty_band": row["uncertainty_band"],
                "selection_reason": row["selection_reason"],
                "active_learning_rank": row["active_learning_rank"],
                "human_observation": "",
                "reviewed_label": "",
                "reviewed_action": "",
                "reviewer_notes": "",
                "review_status": "pending_review",
            }
        )

    return review_rows


def build_summary(review_rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "review_sheet": "stage_01_human_review_sheet_v1",
        "total_review_candidates": len(review_rows),
        "by_source_reference_label": dict(
            sorted(Counter(row["source_reference_label"] for row in review_rows).items())
        ),
        "by_model_prediction": dict(
            sorted(Counter(row["model_prediction"] for row in review_rows).items())
        ),
        "by_selection_reason": dict(
            sorted(Counter(row["selection_reason"] for row in review_rows).items())
        ),
        "by_uncertainty_band": dict(
            sorted(Counter(row["uncertainty_band"] for row in review_rows).items())
        ),
        "review_instruction": (
            "Review each selected image. If the source reference label is correct and belongs "
            "to cardboard, glass, metal, paper, or plastic, set reviewed_label to that class and "
            "reviewed_action to add_to_training. If the image is unclear, mixed, or not one of "
            "the five known classes, set reviewed_action to exclude_or_unknown."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Stage 1 active-learning human review sheet."
    )

    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=Path("ml/outputs/active_learning/stage_01_realwaste_candidate_scores_v1.csv"),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ml/data/manifests/active_learning/stage_01_human_review_sheet_v1.csv"),
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("ml/data/manifests/active_learning/stage_01_human_review_sheet_summary_v1.json"),
    )

    parser.add_argument(
        "--total-limit",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--per-label-minimum-target",
        type=int,
        default=15,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_csv(args.scores_csv)

    selected_rows = select_balanced_candidates(
        rows=rows,
        total_limit=args.total_limit,
        per_label_minimum_target=args.per_label_minimum_target,
    )

    review_rows = build_review_rows(selected_rows)
    summary = build_summary(review_rows)

    fieldnames = list(review_rows[0].keys())

    write_csv(args.output_csv, review_rows, fieldnames)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Human review sheet written: {args.output_csv}")
    print(f"Summary written: {args.summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
