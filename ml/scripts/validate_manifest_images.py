from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate image files referenced by a manifest CSV and create a cleaned CSV."
    )

    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--valid-output-csv", type=Path, required=True)
    parser.add_argument("--invalid-output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_csv(args.input_csv)

    if not rows:
        raise ValueError("Input CSV has no rows.")

    valid_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row["image_path"])
        is_valid, reason = check_image(image_path)

        if is_valid:
            valid_rows.append(row)
        else:
            invalid_row = dict(row)
            invalid_row["invalid_image_reason"] = reason
            invalid_rows.append(invalid_row)

        if index == 1 or index % 500 == 0 or index == len(rows):
            print(
                f"Checked {index}/{len(rows)} | "
                f"valid={len(valid_rows)} | invalid={len(invalid_rows)}",
                flush=True,
            )

    valid_fieldnames = list(rows[0].keys())
    invalid_fieldnames = list(rows[0].keys())

    if "invalid_image_reason" not in invalid_fieldnames:
        invalid_fieldnames.append("invalid_image_reason")

    write_csv(args.valid_output_csv, valid_rows, valid_fieldnames)
    write_csv(args.invalid_output_csv, invalid_rows, invalid_fieldnames)

    summary = {
        "input_csv": str(args.input_csv),
        "valid_output_csv": str(args.valid_output_csv),
        "invalid_output_csv": str(args.invalid_output_csv),
        "total_rows": len(rows),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(invalid_rows),
        "invalid_reason_counts": dict(
            sorted(Counter(row["invalid_image_reason"] for row in invalid_rows).items())
        ),
        "invalid_by_label": dict(
            sorted(Counter(row.get("canonical_label", "") for row in invalid_rows).items())
        ),
        "invalid_by_dataset": dict(
            sorted(Counter(row.get("source_dataset", "") for row in invalid_rows).items())
        ),
        "rule": (
            "Rows with unreadable, missing, or corrupted image files are excluded from model scoring "
            "and training manifests. They are retained separately for auditability."
        ),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
