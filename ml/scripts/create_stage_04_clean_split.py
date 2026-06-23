from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SPLIT_FILES = [
    "known_train_v1.csv",
    "known_val_v1.csv",
    "known_test_v1.csv",
    "unknown_val_v1.csv",
    "unknown_test_v1.csv",
]


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


def check_image(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_file"

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            image.convert("RGB")

        return True, "ok"

    except UnidentifiedImageError:
        return False, "unidentified_image"

    except OSError as error:
        return False, f"os_error: {error}"

    except Exception as error:
        return False, f"unexpected_error: {type(error).__name__}: {error}"


def count_by_key(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(key, "") for row in rows).items()))


def main() -> None:
    input_dir = Path("ml/data/manifests/stages/stage_04_add_trashbox")
    output_dir = Path("ml/data/manifests/stages/stage_04_add_trashbox_clean_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_invalid_rows: list[dict[str, str]] = []
    split_summaries: dict[str, dict[str, object]] = {}

    for split_file in SPLIT_FILES:
        input_path = input_dir / split_file
        output_path = output_dir / split_file

        rows = read_csv(input_path)

        valid_rows: list[dict[str, str]] = []
        invalid_rows: list[dict[str, str]] = []

        print(f"Validating {split_file} | rows={len(rows)}", flush=True)

        for index, row in enumerate(rows, start=1):
            image_path = Path(row["image_path"])
            is_valid, reason = check_image(image_path)

            if is_valid:
                valid_rows.append(row)
            else:
                invalid_row = dict(row)
                invalid_row["split_file"] = split_file
                invalid_row["invalid_image_reason"] = reason
                invalid_rows.append(invalid_row)
                all_invalid_rows.append(invalid_row)

            if index == 1 or index % 1000 == 0 or index == len(rows):
                print(
                    f"{split_file}: checked {index}/{len(rows)} | "
                    f"valid={len(valid_rows)} | invalid={len(invalid_rows)}",
                    flush=True,
                )

        fieldnames = list(rows[0].keys())
        write_csv(output_path, valid_rows, fieldnames)

        split_summaries[split_file] = {
            "input_rows": len(rows),
            "valid_rows": len(valid_rows),
            "invalid_rows": len(invalid_rows),
            "invalid_reason_counts": dict(
                sorted(Counter(row["invalid_image_reason"] for row in invalid_rows).items())
            ),
            "valid_by_label": count_by_key(valid_rows, "canonical_label"),
            "valid_by_dataset": count_by_key(valid_rows, "source_dataset"),
        }

    invalid_output_path = output_dir / "invalid_images_v1.csv"

    invalid_fieldnames = [
        "split_file",
        "image_path",
        "source_dataset",
        "source_label",
        "canonical_label",
        "split_role",
        "invalid_image_reason",
    ]

    normalized_invalid_rows = [
        {field: row.get(field, "") for field in invalid_fieldnames}
        for row in all_invalid_rows
    ]

    write_csv(invalid_output_path, normalized_invalid_rows, invalid_fieldnames)

    known_train = read_csv(output_dir / "known_train_v1.csv")
    known_val = read_csv(output_dir / "known_val_v1.csv")
    known_test = read_csv(output_dir / "known_test_v1.csv")
    unknown_val = read_csv(output_dir / "unknown_val_v1.csv")
    unknown_test = read_csv(output_dir / "unknown_test_v1.csv")

    summary = {
        "stage": "stage_04_add_trashbox_clean_v1",
        "source_stage": "stage_04_add_trashbox",
        "split_summaries": split_summaries,
        "total_invalid_images_removed": len(all_invalid_rows),
        "invalid_images_file": str(invalid_output_path),
        "known_train_rows": len(known_train),
        "known_val_rows": len(known_val),
        "known_test_rows": len(known_test),
        "unknown_val_rows": len(unknown_val),
        "unknown_test_rows": len(unknown_test),
        "known_train_by_label": count_by_key(known_train, "canonical_label"),
        "known_train_by_dataset": count_by_key(known_train, "source_dataset"),
        "rule": (
            "Stage 4 clean split excludes unreadable, missing, or corrupted image files "
            "from all Stage 4 split manifests before training and evaluation."
        ),
    }

    summary_path = output_dir / "stage_summary_v1.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
