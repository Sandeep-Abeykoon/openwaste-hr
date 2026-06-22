from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_by_label(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[row["canonical_label"]].append(row)

    return dict(grouped)


def split_known_rows(
    *,
    rows: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("Known split ratios must add up to 1.0")

    rng = random.Random(seed)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for label, label_rows in grouped_by_label(rows).items():
        shuffled = label_rows[:]
        rng.shuffle(shuffled)

        total = len(shuffled)

        if total < 3:
            raise ValueError(
                f"Not enough images for known label '{label}'. "
                f"Need at least 3, found {total}."
            )

        train_count = max(1, int(total * train_ratio))
        val_count = max(1, int(total * val_ratio))

        if train_count + val_count >= total:
            train_count = max(1, total - 2)
            val_count = 1

        train_rows.extend(shuffled[:train_count])
        val_rows.extend(shuffled[train_count : train_count + val_count])
        test_rows.extend(shuffled[train_count + val_count :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    return train_rows, val_rows, test_rows


def split_unknown_rows(
    *,
    rows: list[dict[str, str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("Unknown validation ratio must be between 0 and 1")

    rng = random.Random(seed)

    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for label, label_rows in grouped_by_label(rows).items():
        shuffled = label_rows[:]
        rng.shuffle(shuffled)

        total = len(shuffled)

        if total < 2:
            raise ValueError(
                f"Not enough images for unknown label '{label}'. "
                f"Need at least 2, found {total}."
            )

        val_count = max(1, int(total * val_ratio))

        if val_count >= total:
            val_count = total - 1

        val_rows.extend(shuffled[:val_count])
        test_rows.extend(shuffled[val_count:])

    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    return val_rows, test_rows


def add_split_column(rows: list[dict[str, str]], split_name: str) -> list[dict[str, str]]:
    updated_rows: list[dict[str, str]] = []

    for row in rows:
        updated = dict(row)
        updated["split"] = split_name
        updated_rows.append(updated)

    return updated_rows


def count_by_label(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["canonical_label"] for row in rows).items()))


def count_by_dataset(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["source_dataset"] for row in rows).items()))


def assert_no_overlap(split_rows: dict[str, list[dict[str, str]]]) -> None:
    seen: dict[str, str] = {}

    for split_name, rows in split_rows.items():
        for row in rows:
            image_path = row["image_path"]

            if image_path in seen:
                raise ValueError(
                    f"Image path appears in multiple splits: {image_path} "
                    f"({seen[image_path]} and {split_name})"
                )

            seen[image_path] = split_name


def build_summary(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    summary: dict[str, object] = {}

    for split_name, rows in split_rows.items():
        summary[split_name] = {
            "total": len(rows),
            "by_label": count_by_label(rows),
            "by_dataset": count_by_dataset(rows),
        }

    return summary


def create_splits(
    *,
    input_dir: Path,
    output_dir: Path,
    known_train_ratio: float,
    known_val_ratio: float,
    known_test_ratio: float,
    unknown_val_ratio: float,
    seed: int,
) -> None:
    known_manifest_path = input_dir / "known_manifest_v1.csv"
    unknown_manifest_path = input_dir / "unknown_manifest_v1.csv"

    known_rows = read_csv(known_manifest_path)
    unknown_rows = read_csv(unknown_manifest_path)

    if not known_rows:
        raise ValueError("known_manifest_v1.csv is empty. Add datasets and run manifest generation first.")

    if not unknown_rows:
        print("WARNING: unknown_manifest_v1.csv is empty. Unknown splits will be empty.")

    known_train, known_val, known_test = split_known_rows(
        rows=known_rows,
        train_ratio=known_train_ratio,
        val_ratio=known_val_ratio,
        test_ratio=known_test_ratio,
        seed=seed,
    )

    if unknown_rows:
        unknown_val, unknown_test = split_unknown_rows(
            rows=unknown_rows,
            val_ratio=unknown_val_ratio,
            seed=seed,
        )
    else:
        unknown_val = []
        unknown_test = []

    split_rows = {
        "known_train": add_split_column(known_train, "known_train"),
        "known_val": add_split_column(known_val, "known_val"),
        "known_test": add_split_column(known_test, "known_test"),
        "unknown_val": add_split_column(unknown_val, "unknown_val"),
        "unknown_test": add_split_column(unknown_test, "unknown_test"),
    }

    assert_no_overlap(split_rows)

    fieldnames = list(known_rows[0].keys())

    if "split" not in fieldnames:
        fieldnames.append("split")

    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "known_train_v1.csv", split_rows["known_train"], fieldnames)
    write_csv(output_dir / "known_val_v1.csv", split_rows["known_val"], fieldnames)
    write_csv(output_dir / "known_test_v1.csv", split_rows["known_test"], fieldnames)
    write_csv(output_dir / "unknown_val_v1.csv", split_rows["unknown_val"], fieldnames)
    write_csv(output_dir / "unknown_test_v1.csv", split_rows["unknown_test"], fieldnames)

    summary = build_summary(split_rows)

    (output_dir / "split_summary_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Dataset splits written successfully:")
    print(f"- {output_dir / 'known_train_v1.csv'}")
    print(f"- {output_dir / 'known_val_v1.csv'}")
    print(f"- {output_dir / 'known_test_v1.csv'}")
    print(f"- {output_dir / 'unknown_val_v1.csv'}")
    print(f"- {output_dir / 'unknown_test_v1.csv'}")
    print(f"- {output_dir / 'split_summary_v1.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OpenWaste-HR known and unknown dataset splits."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("ml/data/manifests"),
        help="Folder containing known_manifest_v1.csv and unknown_manifest_v1.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/data/manifests"),
        help="Folder where split CSV files will be written.",
    )

    parser.add_argument(
        "--known-train-ratio",
        type=float,
        default=0.70,
        help="Training ratio for known classes.",
    )

    parser.add_argument(
        "--known-val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio for known classes.",
    )

    parser.add_argument(
        "--known-test-ratio",
        type=float,
        default=0.15,
        help="Test ratio for known classes.",
    )

    parser.add_argument(
        "--unknown-val-ratio",
        type=float,
        default=0.50,
        help="Validation ratio for unknown classes. Remaining unknowns go to unknown_test.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    create_splits(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        known_train_ratio=args.known_train_ratio,
        known_val_ratio=args.known_val_ratio,
        known_test_ratio=args.known_test_ratio,
        unknown_val_ratio=args.unknown_val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
