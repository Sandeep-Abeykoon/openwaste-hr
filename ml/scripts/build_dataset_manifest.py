from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class LabelMapping:
    source_dataset: str
    source_label: str
    canonical_label: str
    split_role: str
    include_in_known_training: bool
    notes: str


def normalize_label(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
    )


def bool_from_text(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def read_label_mapping(mapping_path: Path) -> dict[str, dict[str, LabelMapping]]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    mappings: dict[str, dict[str, LabelMapping]] = defaultdict(dict)

    with mapping_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "source_dataset",
            "source_label",
            "canonical_label",
            "split_role",
            "include_in_known_training",
            "notes",
        }

        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Mapping file is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            source_dataset = row["source_dataset"].strip()
            source_label = row["source_label"].strip()

            mapping = LabelMapping(
                source_dataset=source_dataset,
                source_label=source_label,
                canonical_label=row["canonical_label"].strip(),
                split_role=row["split_role"].strip(),
                include_in_known_training=bool_from_text(
                    row["include_in_known_training"]
                ),
                notes=row["notes"].strip(),
            )

            mappings[source_dataset][normalize_label(source_label)] = mapping

    return mappings


def iter_image_files(dataset_root: Path) -> Iterable[Path]:
    if not dataset_root.exists():
        return []

    return (
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_source_label_mapping(
    *,
    image_path: Path,
    dataset_root: Path,
    dataset_mappings: dict[str, LabelMapping],
) -> LabelMapping | None:
    relative_parts = image_path.relative_to(dataset_root).parts[:-1]

    for part in relative_parts:
        normalized_part = normalize_label(part)
        if normalized_part in dataset_mappings:
            return dataset_mappings[normalized_part]

    return None


def path_as_posix(path: Path) -> str:
    return path.as_posix()


def build_manifest(
    *,
    raw_root: Path,
    mapping_path: Path,
    output_dir: Path,
    dataset_names: list[str],
) -> None:
    mappings = read_label_mapping(mapping_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "public_dataset_manifest_v1.csv"
    known_manifest_path = output_dir / "known_manifest_v1.csv"
    unknown_manifest_path = output_dir / "unknown_manifest_v1.csv"
    excluded_manifest_path = output_dir / "excluded_manifest_v1.csv"
    unmapped_manifest_path = output_dir / "unmapped_manifest_v1.csv"
    summary_path = output_dir / "manifest_summary_v1.json"

    manifest_rows: list[dict[str, str]] = []
    known_rows: list[dict[str, str]] = []
    unknown_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    unmapped_rows: list[dict[str, str]] = []

    for dataset_name in dataset_names:
        dataset_root = raw_root / dataset_name
        dataset_mappings = mappings.get(dataset_name, {})

        if not dataset_root.exists():
            unmapped_rows.append(
                {
                    "source_dataset": dataset_name,
                    "image_path": "",
                    "source_label": "",
                    "canonical_label": "missing_dataset_folder",
                    "split_role": "missing",
                    "include_in_known_training": "false",
                    "notes": f"Dataset folder does not exist: {dataset_root}",
                }
            )
            continue

        for image_path in iter_image_files(dataset_root):
            mapping = find_source_label_mapping(
                image_path=image_path,
                dataset_root=dataset_root,
                dataset_mappings=dataset_mappings,
            )

            if mapping is None:
                relative_parent = image_path.relative_to(dataset_root).parent.as_posix()
                row = {
                    "source_dataset": dataset_name,
                    "image_path": path_as_posix(image_path),
                    "source_label": relative_parent,
                    "canonical_label": "unmapped",
                    "split_role": "unmapped",
                    "include_in_known_training": "false",
                    "notes": "No matching source label found in source_label_mapping.csv",
                }
                unmapped_rows.append(row)
                manifest_rows.append(row)
                continue

            row = {
                "source_dataset": dataset_name,
                "image_path": path_as_posix(image_path),
                "source_label": mapping.source_label,
                "canonical_label": mapping.canonical_label,
                "split_role": mapping.split_role,
                "include_in_known_training": str(
                    mapping.include_in_known_training
                ).lower(),
                "notes": mapping.notes,
            }

            manifest_rows.append(row)

            if mapping.split_role == "known" and mapping.include_in_known_training:
                known_rows.append(row)
            elif mapping.split_role == "unknown":
                unknown_rows.append(row)
            elif mapping.split_role == "exclude":
                excluded_rows.append(row)
            else:
                unmapped_rows.append(row)

    fieldnames = [
        "source_dataset",
        "image_path",
        "source_label",
        "canonical_label",
        "split_role",
        "include_in_known_training",
        "notes",
    ]

    write_csv(manifest_path, manifest_rows, fieldnames)
    write_csv(known_manifest_path, known_rows, fieldnames)
    write_csv(unknown_manifest_path, unknown_rows, fieldnames)
    write_csv(excluded_manifest_path, excluded_rows, fieldnames)
    write_csv(unmapped_manifest_path, unmapped_rows, fieldnames)

    summary = build_summary(
        manifest_rows=manifest_rows,
        known_rows=known_rows,
        unknown_rows=unknown_rows,
        excluded_rows=excluded_rows,
        unmapped_rows=unmapped_rows,
    )

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Manifest written: {manifest_path}")
    print(f"Known manifest written: {known_manifest_path}")
    print(f"Unknown manifest written: {unknown_manifest_path}")
    print(f"Excluded manifest written: {excluded_manifest_path}")
    print(f"Unmapped manifest written: {unmapped_manifest_path}")
    print(f"Summary written: {summary_path}")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    manifest_rows: list[dict[str, str]],
    known_rows: list[dict[str, str]],
    unknown_rows: list[dict[str, str]],
    excluded_rows: list[dict[str, str]],
    unmapped_rows: list[dict[str, str]],
) -> dict[str, object]:
    by_dataset = Counter(row["source_dataset"] for row in manifest_rows)
    by_split_role = Counter(row["split_role"] for row in manifest_rows)
    by_canonical_label = Counter(row["canonical_label"] for row in manifest_rows)

    known_by_label = Counter(row["canonical_label"] for row in known_rows)
    unknown_by_label = Counter(row["canonical_label"] for row in unknown_rows)

    return {
        "total_images": len(manifest_rows),
        "known_images": len(known_rows),
        "unknown_images": len(unknown_rows),
        "excluded_images": len(excluded_rows),
        "unmapped_or_missing_entries": len(unmapped_rows),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_split_role": dict(sorted(by_split_role.items())),
        "by_canonical_label": dict(sorted(by_canonical_label.items())),
        "known_by_label": dict(sorted(known_by_label.items())),
        "unknown_by_label": dict(sorted(unknown_by_label.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenWaste-HR clean dataset manifests."
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("ml/data/raw"),
        help="Root folder containing raw dataset folders.",
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("ml/configs/source_label_mapping.csv"),
        help="CSV file containing source-to-canonical label mapping.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/data/manifests"),
        help="Output folder for generated manifests.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["trashnet", "realwaste", "garbage_v2", "trashbox"],
        help="Dataset folder names to scan.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_manifest(
        raw_root=args.raw_root,
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        dataset_names=args.datasets,
    )


if __name__ == "__main__":
    main()
