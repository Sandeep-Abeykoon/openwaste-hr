from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "confidence": "Confidence",
    "temperature_scaled_confidence": "Temp-scaled confidence",
    "max_logit": "Max-logit",
    "energy": "Energy",
}


METHOD_DIRECTIONS = {
    "confidence": True,
    "temperature_scaled_confidence": True,
    "max_logit": True,
    "energy": False,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def to_float(value: Any) -> float:
    return float(str(value).strip())


def is_known(row: dict[str, str]) -> bool:
    return str(row.get("split", "")).startswith("known")


def is_unknown(row: dict[str, str]) -> bool:
    return str(row.get("split", "")).startswith("unknown")


def is_correct(row: dict[str, str]) -> bool:
    return str(row.get("true_label", "")).strip() == str(row.get("pred_label", "")).strip()


def save_bar_chart(
    *,
    output_path: Path,
    title: str,
    ylabel: str,
    methods: list[str],
    values: list[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [METHOD_LABELS.get(method, method) for method in methods]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_score_distribution(
    *,
    output_path: Path,
    title: str,
    score_column: str,
    rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    known_scores = [
        to_float(row[score_column])
        for row in rows
        if is_known(row) and row.get(score_column, "") != ""
    ]

    unknown_scores = [
        to_float(row[score_column])
        for row in rows
        if is_unknown(row) and row.get(score_column, "") != ""
    ]

    plt.figure(figsize=(9, 5))
    plt.hist(known_scores, bins=40, alpha=0.65, label="Known")
    plt.hist(unknown_scores, bins=40, alpha=0.65, label="Unknown")
    plt.title(title)
    plt.xlabel(score_column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def calibration_bins(
    *,
    rows: list[dict[str, str]],
    confidence_column: str,
    n_bins: int = 15,
) -> tuple[list[float], list[float], list[int]]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centres: list[float] = []
    accuracies: list[float] = []
    counts: list[int] = []

    for index in range(n_bins):
        low = bin_edges[index]
        high = bin_edges[index + 1]

        if index == n_bins - 1:
            bin_rows = [
                row for row in rows
                if low <= to_float(row[confidence_column]) <= high
            ]
        else:
            bin_rows = [
                row for row in rows
                if low <= to_float(row[confidence_column]) < high
            ]

        if not bin_rows:
            continue

        accuracy = sum(1 for row in bin_rows if is_correct(row)) / len(bin_rows)

        bin_centres.append((low + high) / 2.0)
        accuracies.append(float(accuracy))
        counts.append(len(bin_rows))

    return bin_centres, accuracies, counts


def save_calibration_plot(
    *,
    output_path: Path,
    title: str,
    rows: list[dict[str, str]],
    confidence_column: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    known_test_rows = [
        row for row in rows
        if row.get("split", "") == "known_test"
    ]

    bin_centres, accuracies, counts = calibration_bins(
        rows=known_test_rows,
        confidence_column=confidence_column,
    )

    plt.figure(figsize=(6, 6))
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Perfect calibration")
    plt.plot(bin_centres, accuracies, marker="o", label="Model")
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    counts_path = output_path.with_suffix(".csv")
    with counts_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["bin_confidence", "bin_accuracy", "count"],
        )
        writer.writeheader()

        for bin_confidence, bin_accuracy, count in zip(bin_centres, accuracies, counts):
            writer.writerow(
                {
                    "bin_confidence": bin_confidence,
                    "bin_accuracy": bin_accuracy,
                    "count": count,
                }
            )


def coverage_risk_points(
    *,
    rows: list[dict[str, str]],
    score_column: str,
    higher_is_known: bool,
) -> tuple[list[float], list[float]]:
    known_test_rows = [
        row for row in rows
        if row.get("split", "") == "known_test"
    ]

    scores = np.array(
        [to_float(row[score_column]) for row in known_test_rows],
        dtype=float,
    )

    thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 101))

    coverage_values: list[float] = []
    risk_values: list[float] = []

    for threshold in thresholds:
        if higher_is_known:
            accepted_rows = [
                row for row in known_test_rows
                if to_float(row[score_column]) >= threshold
            ]
        else:
            accepted_rows = [
                row for row in known_test_rows
                if to_float(row[score_column]) <= threshold
            ]

        coverage = len(accepted_rows) / len(known_test_rows)

        if accepted_rows:
            accepted_accuracy = sum(1 for row in accepted_rows if is_correct(row)) / len(accepted_rows)
            risk = 1.0 - accepted_accuracy
        else:
            risk = 1.0

        coverage_values.append(float(coverage))
        risk_values.append(float(risk))

    return coverage_values, risk_values


def save_coverage_risk_plot(
    *,
    output_path: Path,
    normal_rows: list[dict[str, str]],
    temperature_rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for method in ["confidence", "max_logit", "energy"]:
        coverage, risk = coverage_risk_points(
            rows=normal_rows,
            score_column=method,
            higher_is_known=METHOD_DIRECTIONS[method],
        )

        plt.plot(
            coverage,
            risk,
            marker=".",
            label=METHOD_LABELS[method],
        )

    coverage, risk = coverage_risk_points(
        rows=temperature_rows,
        score_column="temperature_scaled_confidence",
        higher_is_known=True,
    )

    plt.plot(
        coverage,
        risk,
        marker=".",
        label=METHOD_LABELS["temperature_scaled_confidence"],
    )

    plt.title("Coverage-risk curve on known test samples")
    plt.xlabel("Known coverage")
    plt.ylabel("Selective risk")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots for reject-option/open-set evaluation."
    )

    parser.add_argument(
        "--open-set-dir",
        type=Path,
        default=Path("ml/outputs/reject_option/stage_04_final_open_set_v1"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/results/figures"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    normal_predictions = read_csv(args.open_set_dir / "open_set_predictions_v1.csv")
    normal_metrics = read_csv(args.open_set_dir / "open_set_metrics_v1.csv")

    temperature_predictions = read_csv(args.open_set_dir / "temperature_scaled_predictions_v1.csv")
    temperature_metrics = read_csv(args.open_set_dir / "temperature_scaled_metrics_v1.csv")

    combined_metrics = normal_metrics + temperature_metrics

    method_order = [
        "confidence",
        "temperature_scaled_confidence",
        "max_logit",
        "energy",
    ]

    metric_lookup = {
        row["score_method"]: row
        for row in combined_metrics
    }

    metric_specs = [
        (
            "test_auroc_known_vs_unknown",
            "Reject-option AUROC comparison",
            "AUROC",
            "reject_option_auroc_comparison_v1.png",
        ),
        (
            "test_known_coverage",
            "Known coverage comparison",
            "Known coverage",
            "reject_option_known_coverage_comparison_v1.png",
        ),
        (
            "test_unknown_rejection_rate",
            "Unknown rejection comparison",
            "Unknown rejection rate",
            "reject_option_unknown_rejection_comparison_v1.png",
        ),
        (
            "test_known_accuracy_accepted",
            "Accepted-known accuracy comparison",
            "Accepted-known accuracy",
            "reject_option_accepted_known_accuracy_comparison_v1.png",
        ),
    ]

    for metric_name, title, ylabel, filename in metric_specs:
        values = [
            to_float(metric_lookup[method][metric_name])
            for method in method_order
        ]

        save_bar_chart(
            output_path=args.output_dir / filename,
            title=title,
            ylabel=ylabel,
            methods=method_order,
            values=values,
        )

    save_score_distribution(
        output_path=args.output_dir / "reject_option_confidence_distribution_v1.png",
        title="Raw confidence distribution: known vs unknown",
        score_column="confidence",
        rows=normal_predictions,
    )

    save_score_distribution(
        output_path=args.output_dir / "reject_option_temperature_scaled_confidence_distribution_v1.png",
        title="Temperature-scaled confidence distribution: known vs unknown",
        score_column="temperature_scaled_confidence",
        rows=temperature_predictions,
    )

    save_score_distribution(
        output_path=args.output_dir / "reject_option_max_logit_distribution_v1.png",
        title="Max-logit distribution: known vs unknown",
        score_column="max_logit",
        rows=normal_predictions,
    )

    save_score_distribution(
        output_path=args.output_dir / "reject_option_energy_distribution_v1.png",
        title="Energy distribution: known vs unknown",
        score_column="energy",
        rows=normal_predictions,
    )

    save_calibration_plot(
        output_path=args.output_dir / "reject_option_calibration_before_temperature_v1.png",
        title="Calibration before temperature scaling",
        rows=normal_predictions,
        confidence_column="confidence",
    )

    save_calibration_plot(
        output_path=args.output_dir / "reject_option_calibration_after_temperature_v1.png",
        title="Calibration after temperature scaling",
        rows=temperature_predictions,
        confidence_column="temperature_scaled_confidence",
    )

    save_coverage_risk_plot(
        output_path=args.output_dir / "reject_option_coverage_risk_curve_v1.png",
        normal_rows=normal_predictions,
        temperature_rows=temperature_predictions,
    )

    print(f"Reject-option plots written to: {args.output_dir}")


if __name__ == "__main__":
    main()
