from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


STAGES = {
    "stage_01_trashnet": {
        "description": "TrashNet-only 5-class baseline.",
        "known_datasets": {"trashnet"},
        "unknown_datasets": {"realwaste", "garbage_v2"},
    },
    "stage_02_trashnet_realwaste": {
        "description": "TrashNet plus RealWaste known-class expansion.",
        "known_datasets": {"trashnet", "realwaste"},
        "unknown_datasets": {"realwaste", "garbage_v2"},
    },
    "stage_03_add_garbage_v2": {
        "description": "TrashNet, RealWaste, and Garbage Classification V2 expansion.",
        "known_datasets": {"trashnet", "realwaste", "garbage_v2"},
        "unknown_datasets": {"realwaste", "garbage_v2"},
    },
    "stage_04_add_trashbox": {
        "description": "Final combined public known dataset with TrashBox added.",
        "known_datasets": {"trashnet", "realwaste", "garbage_v2", "trashbox"},
        "unknown_datasets": {"realwaste", "garbage_v2"},
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required split file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_by_dataset(
    rows: list[dict[str, str]],
    allowed_datasets: set[str],
) -> list[dict[str, str]]:
    return [row for row in rows if row["source_dataset"] in allowed_datasets]


def count_by_key(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def build_stage_splits(input_dir: Path, output_root: Path) -> None:
    known_train = read_csv(input_dir / "known_train_v1.csv")
    known_val = read_csv(input_dir / "known_val_v1.csv")
    known_test = read_csv(input_dir / "known_test_v1.csv")
    unknown_val = read_csv(input_dir / "unknown_val_v1.csv")
    unknown_test = read_csv(input_dir / "unknown_test_v1.csv")

    fieldnames = list(known_train[0].keys())

    all_stage_summary: dict[str, object] = {}

    for stage_name, stage_config in STAGES.items():
        stage_dir = output_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_known_train = filter_by_dataset(
            known_train,
            stage_config["known_datasets"],
        )
        stage_known_val = filter_by_dataset(
            known_val,
            stage_config["known_datasets"],
        )
        stage_known_test = filter_by_dataset(
            known_test,
            stage_config["known_datasets"],
        )
        stage_unknown_val = filter_by_dataset(
            unknown_val,
            stage_config["unknown_datasets"],
        )
        stage_unknown_test = filter_by_dataset(
            unknown_test,
            stage_config["unknown_datasets"],
        )

        split_rows = {
            "known_train": stage_known_train,
            "known_val": stage_known_val,
            "known_test": stage_known_test,
            "unknown_val": stage_unknown_val,
            "unknown_test": stage_unknown_test,
        }

        for split_name, rows in split_rows.items():
            write_csv(stage_dir / f"{split_name}_v1.csv", rows, fieldnames)

        stage_summary = {
            "description": stage_config["description"],
            "known_datasets": sorted(stage_config["known_datasets"]),
            "unknown_datasets": sorted(stage_config["unknown_datasets"]),
            "splits": {
                split_name: {
                    "total": len(rows),
                    "by_label": count_by_key(rows, "canonical_label"),
                    "by_dataset": count_by_key(rows, "source_dataset"),
                }
                for split_name, rows in split_rows.items()
            },
        }

        (stage_dir / "stage_summary_v1.json").write_text(
            json.dumps(stage_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        all_stage_summary[stage_name] = stage_summary

        print(f"Created stage splits: {stage_name}")

    (output_root / "all_stage_summary_v1.json").write_text(
        json.dumps(all_stage_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"All stage summary written: {output_root / 'all_stage_summary_v1.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stage-specific OpenWaste-HR split files."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("ml/data/manifests"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ml/data/manifests/stages"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_stage_splits(args.input_dir, args.output_root)


if __name__ == "__main__":
    main()
