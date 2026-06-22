from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


KNOWN_FINE_LABELS = {
    "paper_cardboard": "recyclable",
    "plastic": "recyclable",
    "glass": "recyclable",
    "metal": "recyclable",
    "organic": "organic",
    "residual": "residual",
}

MIN_KNOWN_TRAIN_SAMPLES = 10
MIN_KNOWN_TRAIN_CLASSES = 2


@dataclass(frozen=True)
class AuditedRecord:
    source_file: str
    sample_id: str
    image_path: str
    human_observation: str
    taxonomy_status: str
    reviewed_fine_label: str
    reviewed_coarse_label: str
    recommended_action: str
    active_learning_role: str
    audit_decision: str
    audit_reason: str


def _normalise_key(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _normalise_label(value: str) -> str:
    return _normalise_key(value)


def _normalise_row(row: dict[str, str]) -> dict[str, str]:
    return {_normalise_key(str(key)): str(value).strip() for key, value in row.items()}


def _first_non_empty(row: dict[str, str], candidate_keys: Iterable[str]) -> str:
    normalised = _normalise_row(row)

    for key in candidate_keys:
        value = normalised.get(_normalise_key(key), "").strip()
        if value:
            return value

    return ""


def _contains_any(value: str, keywords: Iterable[str]) -> bool:
    lowered = value.lower()
    return any(keyword in lowered for keyword in keywords)


def has_review_signal(row: dict[str, str]) -> bool:
    """Return True when a row appears to contain a human/manual review decision."""

    review_values = [
        _first_non_empty(row, ["human_observation", "human_description", "reviewer_observation"]),
        _first_non_empty(row, ["taxonomy_status"]),
        _first_non_empty(row, ["reviewed_fine_label", "human_fine_label"]),
        _first_non_empty(row, ["reviewed_coarse_label", "human_coarse_label"]),
        _first_non_empty(row, ["recommended_action", "action"]),
        _first_non_empty(row, ["active_learning_v2_role", "active_learning_role", "role"]),
    ]

    ignored_values = {"", "none", "nan", "pending", "todo", "tbd", "to_review"}

    return any(value.strip().lower() not in ignored_values for value in review_values)


def classify_review_record(row: dict[str, str]) -> tuple[str, str]:
    """Classify a reviewed record into an active-learning audit decision."""

    taxonomy_status = _first_non_empty(row, ["taxonomy_status"])
    reviewed_fine_label = _normalise_label(
        _first_non_empty(row, ["reviewed_fine_label", "human_fine_label"])
    )
    recommended_action = _first_non_empty(row, ["recommended_action", "action"])
    active_learning_role = _first_non_empty(
        row, ["active_learning_v2_role", "active_learning_role", "role"]
    )
    human_observation = _first_non_empty(
        row, ["human_observation", "human_description", "reviewer_observation"]
    )

    combined_decision_text = " ".join(
        [
            taxonomy_status,
            reviewed_fine_label,
            recommended_action,
            active_learning_role,
            human_observation,
        ]
    ).lower()

    outside_taxonomy = _contains_any(
        combined_decision_text,
        [
            "outside_current_known_taxonomy",
            "outside taxonomy",
            "outside",
            "unknown",
            "future_class",
            "future class",
            "keep_as_unknown",
        ],
    )

    exclude_or_review = _contains_any(
        combined_decision_text,
        [
            "exclude",
            "review_again",
            "unclear",
            "blur",
            "duplicate",
            "unreadable",
        ],
    )

    known_train_requested = _contains_any(
        combined_decision_text,
        [
            "known_train_candidate",
            "train_candidate",
            "add_to_train",
            "training_candidate",
        ],
    )

    known_eval_requested = _contains_any(
        combined_decision_text,
        [
            "known_eval_candidate",
            "eval_candidate",
            "evaluation_candidate",
            "test_candidate_known",
        ],
    )

    fine_label_is_known = reviewed_fine_label in KNOWN_FINE_LABELS

    if exclude_or_review:
        return (
            "exclude_or_review_again",
            "The record is unclear, duplicated, excluded, or requires another review.",
        )

    if known_train_requested:
        if fine_label_is_known and not outside_taxonomy:
            return (
                "known_train_candidate",
                "The reviewed sample is eligible for active-learning retraining.",
            )

        return (
            "review_needed",
            "The record requests training use but does not have a safe known fine label.",
        )

    if known_eval_requested:
        if fine_label_is_known and not outside_taxonomy:
            return (
                "known_eval_candidate",
                "The reviewed sample belongs to a known class but is marked for evaluation.",
            )

        return (
            "review_needed",
            "The record requests evaluation use but does not have a safe known fine label.",
        )

    if outside_taxonomy:
        return (
            "unknown_or_future_candidate",
            "The reviewed sample is outside the current known taxonomy or marked as unknown/future-class.",
        )

    if fine_label_is_known:
        return (
            "known_train_candidate",
            "The reviewed sample has a valid known fine label and no outside-taxonomy warning.",
        )

    return (
        "review_needed",
        "The record does not contain enough information for a safe active-learning decision.",
    )


def discover_review_files(project_root: Path) -> list[Path]:
    search_root = project_root / "ml" / "data" / "splits"

    if not search_root.exists():
        return []

    candidate_files: list[Path] = []

    for csv_path in search_root.rglob("*.csv"):
        name = csv_path.name.lower()

        if "audit" in name:
            continue

        if any(keyword in name for keyword in ["review", "human", "manual", "active_learning"]):
            candidate_files.append(csv_path)

    return sorted(candidate_files)


def read_review_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return []

        return [row for row in reader if has_review_signal(row)]


def audit_manual_review_records(project_root: Path) -> list[AuditedRecord]:
    audited_records: list[AuditedRecord] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for source_path in discover_review_files(project_root):
        rows = read_review_rows(source_path)

        for row in rows:
            sample_id = _first_non_empty(row, ["sample_id", "id"])
            image_path = _first_non_empty(row, ["image_path", "path"])
            human_observation = _first_non_empty(
                row, ["human_observation", "human_description", "reviewer_observation"]
            )

            dedupe_key = (sample_id, image_path, human_observation)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            audit_decision, audit_reason = classify_review_record(row)

            reviewed_fine_label = _normalise_label(
                _first_non_empty(row, ["reviewed_fine_label", "human_fine_label"])
            )
            reviewed_coarse_label = _normalise_label(
                _first_non_empty(row, ["reviewed_coarse_label", "human_coarse_label"])
            )

            audited_records.append(
                AuditedRecord(
                    source_file=str(source_path.relative_to(project_root)),
                    sample_id=sample_id,
                    image_path=image_path,
                    human_observation=human_observation,
                    taxonomy_status=_first_non_empty(row, ["taxonomy_status"]),
                    reviewed_fine_label=reviewed_fine_label,
                    reviewed_coarse_label=reviewed_coarse_label,
                    recommended_action=_first_non_empty(row, ["recommended_action", "action"]),
                    active_learning_role=_first_non_empty(
                        row, ["active_learning_v2_role", "active_learning_role", "role"]
                    ),
                    audit_decision=audit_decision,
                    audit_reason=audit_reason,
                )
            )

    return audited_records


def write_audit_csv(records: list[AuditedRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_file",
        "sample_id",
        "image_path",
        "human_observation",
        "taxonomy_status",
        "reviewed_fine_label",
        "reviewed_coarse_label",
        "recommended_action",
        "active_learning_role",
        "audit_decision",
        "audit_reason",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "source_file": record.source_file,
                    "sample_id": record.sample_id,
                    "image_path": record.image_path,
                    "human_observation": record.human_observation,
                    "taxonomy_status": record.taxonomy_status,
                    "reviewed_fine_label": record.reviewed_fine_label,
                    "reviewed_coarse_label": record.reviewed_coarse_label,
                    "recommended_action": record.recommended_action,
                    "active_learning_role": record.active_learning_role,
                    "audit_decision": record.audit_decision,
                    "audit_reason": record.audit_reason,
                }
            )


def build_summary(records: list[AuditedRecord]) -> dict[str, object]:
    decision_counts = Counter(record.audit_decision for record in records)

    known_train_classes = Counter(
        record.reviewed_fine_label
        for record in records
        if record.audit_decision == "known_train_candidate"
    )

    invalid_training_records = [
        record
        for record in records
        if record.audit_decision == "review_needed"
        and "train" in f"{record.recommended_action} {record.active_learning_role}".lower()
    ]

    retraining_ready = (
        decision_counts["known_train_candidate"] >= MIN_KNOWN_TRAIN_SAMPLES
        and len(known_train_classes) >= MIN_KNOWN_TRAIN_CLASSES
        and len(invalid_training_records) == 0
    )

    return {
        "total_reviewed_records": len(records),
        "decision_counts": dict(decision_counts),
        "known_train_classes": dict(known_train_classes),
        "invalid_training_records": len(invalid_training_records),
        "retraining_ready": retraining_ready,
    }


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    header = rows[0]
    separator = ["---"] * len(header)
    body = rows[1:]

    all_rows = [header, separator, *body]
    return "\n".join("| " + " | ".join(row) + " |" for row in all_rows)


def write_markdown_report(records: list[AuditedRecord], output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(records)
    decision_counts = summary["decision_counts"]
    known_train_classes = summary["known_train_classes"]

    decision_rows = [["Audit Decision", "Count"]]
    for decision in [
        "known_train_candidate",
        "known_eval_candidate",
        "unknown_or_future_candidate",
        "exclude_or_review_again",
        "review_needed",
    ]:
        decision_rows.append([decision, str(decision_counts.get(decision, 0))])

    class_rows = [["Fine Label", "Known Training Candidate Count"]]
    for fine_label in KNOWN_FINE_LABELS:
        class_rows.append([fine_label, str(known_train_classes.get(fine_label, 0))])

    record_rows = [["Sample ID", "Observation", "Decision", "Reason"]]
    for record in records:
        record_rows.append(
            [
                record.sample_id or "-",
                record.human_observation or "-",
                record.audit_decision,
                record.audit_reason,
            ]
        )

    retraining_ready = bool(summary["retraining_ready"])

    if retraining_ready:
        retraining_decision = (
            "Active learning retraining is ready because there are enough reviewed "
            "known-class samples across enough known classes."
        )
    else:
        retraining_decision = (
            "Active learning retraining is not ready yet. More reviewed known-class "
            "samples are required before fine-tuning the expanded public pretrained model."
        )

    content = f"""# Manual Review Records Audit v1

## Purpose

This report audits the current manual review records for OpenWaste-HR active learning.

The goal is to decide whether the existing reviewed records are ready for active learning retraining or whether they should remain as unknown/future-class evidence.

## Summary

| Metric | Value |
|---|---:|
| total reviewed records | {summary["total_reviewed_records"]} |
| known train candidates | {decision_counts.get("known_train_candidate", 0)} |
| known evaluation candidates | {decision_counts.get("known_eval_candidate", 0)} |
| unknown or future-class candidates | {decision_counts.get("unknown_or_future_candidate", 0)} |
| exclude or review again | {decision_counts.get("exclude_or_review_again", 0)} |
| review needed | {decision_counts.get("review_needed", 0)} |
| invalid training records | {summary["invalid_training_records"]} |
| retraining ready | {str(retraining_ready).lower()} |

## Decision Counts

{_markdown_table(decision_rows)}

## Known Training Candidates by Class

{_markdown_table(class_rows)}

## Reviewed Record Decisions

{_markdown_table(record_rows)}

## Retraining Decision

{retraining_decision}

## Interpretation

Manual review records should only be used for retraining when they clearly belong to the current known taxonomy.

Outside-taxonomy records should not be forced into known training classes. They should remain useful for local unknown evaluation, future-class analysis, and manual-review evidence.

The next step is to complete more manual review records and then rerun this audit.
"""

    output_md.write_text(content, encoding="utf-8")


def write_supervisor_summary(records: list[AuditedRecord], output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(records)
    decision_counts = summary["decision_counts"]
    retraining_ready = bool(summary["retraining_ready"])

    content = f"""# Manual Review Records Audit Summary v1

## Purpose

This stage audited the available manual review records for OpenWaste-HR active learning.

## Result

| Metric | Value |
|---|---:|
| total reviewed records | {summary["total_reviewed_records"]} |
| known train candidates | {decision_counts.get("known_train_candidate", 0)} |
| unknown or future-class candidates | {decision_counts.get("unknown_or_future_candidate", 0)} |
| retraining ready | {str(retraining_ready).lower()} |

## Conclusion

The audit checks whether the project has enough reviewed known-class samples for active learning retraining.

If retraining is not ready, the project should continue collecting and reviewing local samples instead of forcing outside-taxonomy samples into known classes.

This supports the thesis argument that human-in-the-loop active learning must protect dataset quality, not only increase dataset size.
"""

    output_md.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit manual review records for OpenWaste-HR active learning."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ml/outputs/active_learning/manual_review_records_audit_v1.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/results/manual_review_records_audit_v1.md"),
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("docs/supervisor_updates/manual_review_records_audit_summary_v1.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    records = audit_manual_review_records(project_root)

    write_audit_csv(records, project_root / args.output_csv)
    write_markdown_report(records, project_root / args.output_md)
    write_supervisor_summary(records, project_root / args.summary_md)

    summary = build_summary(records)

    print("Manual review records audit complete")
    print(f"Total reviewed records: {summary['total_reviewed_records']}")
    print(
        "Known train candidates: "
        f"{summary['decision_counts'].get('known_train_candidate', 0)}"
    )
    print(
        "Unknown/future-class candidates: "
        f"{summary['decision_counts'].get('unknown_or_future_candidate', 0)}"
    )
    print(f"Retraining ready: {str(summary['retraining_ready']).lower()}")


if __name__ == "__main__":
    main()