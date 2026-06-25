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


def count_by_key(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def main() -> None:
    stage_02_dir = Path("ml/data/manifests/stages/stage_02_trashnet_realwaste")
    output_dir = Path("ml/data/manifests/stages/stage_02_trashnet_realwaste_active_learning_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_train = read_csv(stage_02_dir / "known_train_v1.csv")
    base_val = read_csv(stage_02_dir / "known_val_v1.csv")
    base_test = read_csv(stage_02_dir / "known_test_v1.csv")
    unknown_val = read_csv(stage_02_dir / "unknown_val_v1.csv")
    unknown_test = read_csv(stage_02_dir / "unknown_test_v1.csv")

    reviewed_additions = read_csv(
        Path("ml/data/manifests/active_learning/stage_02_reviewed_known_additions_v1.csv")
    )

    existing_paths = {row["image_path"] for row in base_train}
    clean_additions = [
        row for row in reviewed_additions
        if row["image_path"] not in existing_paths
    ]

    active_train = base_train + clean_additions

    fieldnames = list(base_train[0].keys())

    for extra_column in [
        "active_learning_stage",
        "review_id",
        "human_observation",
        "reviewed_label",
        "reviewed_action",
        "reviewer_notes",
    ]:
        if extra_column not in fieldnames:
            fieldnames.append(extra_column)

    normalized_train = []

    for row in active_train:
        updated = {field: row.get(field, "") for field in fieldnames}
        normalized_train.append(updated)

    split_files = [
        ("known_train_v1.csv", normalized_train),
        ("known_val_v1.csv", base_val),
        ("known_test_v1.csv", base_test),
        ("unknown_val_v1.csv", unknown_val),
        ("unknown_test_v1.csv", unknown_test),
    ]

    for filename, rows in split_files:
        normalized_rows = [
            {field: row.get(field, "") for field in fieldnames}
            for row in rows
        ]
        write_csv(output_dir / filename, normalized_rows, fieldnames)

    summary = {
        "stage": "stage_02_trashnet_realwaste_active_learning_v1",
        "base_training_rows": len(base_train),
        "reviewed_additions_available": len(reviewed_additions),
        "reviewed_additions_added": len(clean_additions),
        "final_training_rows": len(normalized_train),
        "known_val_rows": len(base_val),
        "known_test_rows": len(base_test),
        "unknown_val_rows": len(unknown_val),
        "unknown_test_rows": len(unknown_test),
        "training_by_label": count_by_key(normalized_train, "canonical_label"),
        "training_by_dataset": count_by_key(normalized_train, "source_dataset"),
        "rule": (
            "Stage 2 active-learning retraining uses TrashNet + RealWaste training data "
            "plus only human-reviewed Garbage V2 known-class additions."
        ),
    }

    (output_dir / "stage_summary_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
