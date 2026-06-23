from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    stage_03_train_path = Path(
        "ml/data/manifests/stages/stage_03_add_garbage_v2/known_train_v1.csv"
    )

    output_path = Path(
        "ml/data/manifests/active_learning/stage_02_garbage_v2_candidate_pool_v1.csv"
    )

    summary_path = Path(
        "ml/data/manifests/active_learning/stage_02_garbage_v2_candidate_pool_summary_v1.json"
    )

    rows = read_csv(stage_03_train_path)

    candidate_rows: list[dict[str, str]] = []

    for row in rows:
        if row["source_dataset"] != "garbage_v2":
            continue

        updated = dict(row)
        updated["active_learning_stage"] = "stage_02_after_trashnet_realwaste"
        updated["candidate_pool_role"] = "garbage_v2_known_candidate_pool"
        updated["review_status"] = "pending_model_scoring"
        updated["human_review_label"] = ""
        updated["human_review_action"] = ""
        updated["reviewer_notes"] = ""
        candidate_rows.append(updated)

    if not candidate_rows:
        raise ValueError("No Garbage V2 candidate rows found for Stage 2 active learning.")

    fieldnames = list(candidate_rows[0].keys())

    write_csv(output_path, candidate_rows, fieldnames)

    summary = {
        "active_learning_stage": "stage_02_after_trashnet_realwaste",
        "candidate_pool": "Garbage V2 known-class training subset",
        "total_candidates": len(candidate_rows),
        "by_label": dict(sorted(Counter(row["canonical_label"] for row in candidate_rows).items())),
        "by_dataset": dict(sorted(Counter(row["source_dataset"] for row in candidate_rows).items())),
        "rule": (
            "These candidates are used for model scoring and human review. "
            "Only human-confirmed known-class samples may be added to active-learning retraining."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Candidate pool written: {output_path}")
    print(f"Summary written: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
