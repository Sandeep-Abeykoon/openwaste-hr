from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


DEFAULT_THRESHOLD = 0.6314586412215439


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    text = str(value).strip()

    if text == "":
        return default

    return float(text)


def to_bool(value: Any) -> bool:
    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def is_correct(row: dict[str, Any]) -> bool:
    if "is_correct" in row and str(row.get("is_correct", "")).strip() != "":
        return to_bool(row.get("is_correct"))

    return str(row.get("true_label", "")).strip() == str(row.get("pred_label", "")).strip()


def get_score(row: dict[str, Any]) -> float:
    for key in [
        "fusion_v2_knownness_score",
        "fusion_knownness_score",
        "knownness_score",
    ]:
        if key in row and str(row.get(key, "")).strip() != "":
            return to_float(row[key])

    raise KeyError(
        "Could not find fusion knownness score column. "
        "Expected fusion_v2_knownness_score."
    )


def get_test_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    known_rows = [
        row for row in rows
        if str(row.get("split", "")).strip() == "known_test"
    ]

    unknown_rows = [
        row for row in rows
        if str(row.get("split", "")).strip() == "unknown_test"
    ]

    if not known_rows:
        raise ValueError("No known_test rows found.")

    if not unknown_rows:
        raise ValueError("No unknown_test rows found.")

    return known_rows, unknown_rows


def build_arrays(
    known_rows: list[dict[str, str]],
    unknown_rows: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    known_scores = np.array([get_score(row) for row in known_rows], dtype=float)
    unknown_scores = np.array([get_score(row) for row in unknown_rows], dtype=float)

    known_correct = np.array([is_correct(row) for row in known_rows], dtype=bool)

    y_true = np.concatenate(
        [
            np.ones(len(known_scores), dtype=int),
            np.zeros(len(unknown_scores), dtype=int),
        ]
    )

    y_score = np.concatenate([known_scores, unknown_scores])
    y_score = np.clip(y_score, 0.0, 1.0)

    return {
        "known_scores": known_scores,
        "unknown_scores": unknown_scores,
        "known_correct": known_correct,
        "y_true": y_true,
        "y_score": y_score,
    }


def compute_ece(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int,
) -> tuple[float, list[dict[str, Any]]]:
    y_score = np.clip(y_score, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows: list[dict[str, Any]] = []

    total = len(y_true)

    for bin_index in range(n_bins):
        lower = float(bin_edges[bin_index])
        upper = float(bin_edges[bin_index + 1])

        if bin_index == n_bins - 1:
            mask = (y_score >= lower) & (y_score <= upper)
        else:
            mask = (y_score >= lower) & (y_score < upper)

        count = int(np.sum(mask))

        if count == 0:
            bin_rows.append(
                {
                    "bin_index": bin_index,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "weight": 0.0,
                    "mean_confidence": "",
                    "empirical_known_rate": "",
                    "absolute_gap": "",
                }
            )
            continue

        mean_confidence = float(np.mean(y_score[mask]))
        empirical_known_rate = float(np.mean(y_true[mask]))
        absolute_gap = abs(empirical_known_rate - mean_confidence)
        weight = count / total

        ece += weight * absolute_gap

        bin_rows.append(
            {
                "bin_index": bin_index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "weight": float(weight),
                "mean_confidence": mean_confidence,
                "empirical_known_rate": empirical_known_rate,
                "absolute_gap": float(absolute_gap),
            }
        )

    return float(ece), bin_rows


def compute_fixed_threshold_metrics(
    *,
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_correct: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_bins: int,
) -> dict[str, float]:
    known_accept = known_scores >= threshold
    unknown_accept = unknown_scores >= threshold

    known_coverage = float(np.mean(known_accept))
    unknown_acceptance_rate = float(np.mean(unknown_accept))
    unknown_rejection_rate = 1.0 - unknown_acceptance_rate

    if np.sum(known_accept) > 0:
        accepted_known_accuracy = float(np.mean(known_correct[known_accept]))
    else:
        accepted_known_accuracy = 0.0

    auroc = float(roc_auc_score(y_true, y_score))

    pauc_fpr_0_05 = float(
        roc_auc_score(
            y_true,
            y_score,
            max_fpr=0.05,
        )
    )

    pauc_fpr_0_10 = float(
        roc_auc_score(
            y_true,
            y_score,
            max_fpr=0.10,
        )
    )

    brier = float(brier_score_loss(y_true, y_score))
    ece, _ = compute_ece(y_true=y_true, y_score=y_score, n_bins=n_bins)

    return {
        "known_coverage": known_coverage,
        "known_rejection_rate": 1.0 - known_coverage,
        "unknown_rejection_rate": unknown_rejection_rate,
        "false_acceptance_rate": unknown_acceptance_rate,
        "accepted_known_accuracy": accepted_known_accuracy,
        "selective_risk": 1.0 - accepted_known_accuracy,
        "binary_balanced_accuracy": (known_coverage + unknown_rejection_rate) / 2.0,
        "auroc_known_vs_unknown": auroc,
        "standardized_pauc_fpr_0_05": pauc_fpr_0_05,
        "standardized_pauc_fpr_0_10": pauc_fpr_0_10,
        "brier_score": brier,
        "ece": ece,
    }


def bootstrap_confidence_intervals(
    *,
    arrays: dict[str, np.ndarray],
    threshold: float,
    n_bins: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)

    known_scores = arrays["known_scores"]
    unknown_scores = arrays["unknown_scores"]
    known_correct = arrays["known_correct"]

    bootstrap_values: dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        known_indices = rng.integers(0, len(known_scores), size=len(known_scores))
        unknown_indices = rng.integers(0, len(unknown_scores), size=len(unknown_scores))

        sampled_known_scores = known_scores[known_indices]
        sampled_unknown_scores = unknown_scores[unknown_indices]
        sampled_known_correct = known_correct[known_indices]

        y_true = np.concatenate(
            [
                np.ones(len(sampled_known_scores), dtype=int),
                np.zeros(len(sampled_unknown_scores), dtype=int),
            ]
        )

        y_score = np.concatenate([sampled_known_scores, sampled_unknown_scores])
        y_score = np.clip(y_score, 0.0, 1.0)

        metrics = compute_fixed_threshold_metrics(
            known_scores=sampled_known_scores,
            unknown_scores=sampled_unknown_scores,
            known_correct=sampled_known_correct,
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            n_bins=n_bins,
        )

        for metric_name, value in metrics.items():
            bootstrap_values.setdefault(metric_name, []).append(float(value))

    confidence_intervals: dict[str, dict[str, float]] = {}

    for metric_name, values in bootstrap_values.items():
        values_array = np.array(values, dtype=float)

        confidence_intervals[metric_name] = {
            "bootstrap_mean": float(np.mean(values_array)),
            "ci_lower_95": float(np.percentile(values_array, 2.5)),
            "ci_upper_95": float(np.percentile(values_array, 97.5)),
            "bootstrap_std": float(np.std(values_array, ddof=1)),
        }

    return confidence_intervals


def create_roc_curve_rows(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> list[dict[str, Any]]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    rows: list[dict[str, Any]] = []

    for index, (fpr_value, tpr_value, threshold) in enumerate(zip(fpr, tpr, thresholds)):
        rows.append(
            {
                "index": index,
                "far_fpr": float(fpr_value),
                "known_tpr": float(tpr_value),
                "unknown_rejection_rate": float(1.0 - fpr_value),
                "threshold": float(threshold),
            }
        )

    return rows


def create_calibration_plot(
    *,
    ece_bin_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values: list[float] = []
    y_values: list[float] = []

    for row in ece_bin_rows:
        if row["count"] == 0:
            continue

        x_values.append(float(row["mean_confidence"]))
        y_values.append(float(row["empirical_known_rate"]))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    if x_values and y_values:
        plt.plot(x_values, y_values, marker="o", label="Fusion Gate v2")

    plt.xlabel("Mean predicted knownness")
    plt.ylabel("Empirical known rate")
    plt.title("Fusion Gate v2 Calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_low_far_roc_plot(
    *,
    roc_rows: list[dict[str, Any]],
    output_path: Path,
    max_far: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [
        float(row["far_fpr"])
        for row in roc_rows
        if float(row["far_fpr"]) <= max_far
    ]

    y_values = [
        float(row["known_tpr"])
        for row in roc_rows
        if float(row["far_fpr"]) <= max_far
    ]

    plt.figure(figsize=(7, 5))

    if x_values and y_values:
        plt.plot(x_values, y_values, marker="o")

    plt.xlabel("False acceptance rate")
    plt.ylabel("Known true positive rate")
    plt.title(f"Low-FAR ROC Region, FAR <= {max_far}")
    plt.xlim(0, max_far)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Fusion Gate v2 statistical metrics."
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "ml/outputs/fusion_gate/stage_04_fusion_gate_v2_mahalanobis/"
            "fusion_gate_v2_predictions.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "ml/outputs/fusion_gate/stage_04_fusion_gate_v2_statistical_eval"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_csv(args.predictions)
    known_rows, unknown_rows = get_test_rows(rows)

    arrays = build_arrays(known_rows, unknown_rows)

    point_metrics = compute_fixed_threshold_metrics(
        known_scores=arrays["known_scores"],
        unknown_scores=arrays["unknown_scores"],
        known_correct=arrays["known_correct"],
        y_true=arrays["y_true"],
        y_score=arrays["y_score"],
        threshold=args.threshold,
        n_bins=args.n_bins,
    )

    confidence_intervals = bootstrap_confidence_intervals(
        arrays=arrays,
        threshold=args.threshold,
        n_bins=args.n_bins,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    ece_value, ece_bin_rows = compute_ece(
        y_true=arrays["y_true"],
        y_score=arrays["y_score"],
        n_bins=args.n_bins,
    )

    roc_rows = create_roc_curve_rows(
        y_true=arrays["y_true"],
        y_score=arrays["y_score"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []

    for metric_name, point_value in point_metrics.items():
        ci = confidence_intervals[metric_name]

        metric_rows.append(
            {
                "metric": metric_name,
                "point_estimate": float(point_value),
                "bootstrap_mean": ci["bootstrap_mean"],
                "ci_lower_95": ci["ci_lower_95"],
                "ci_upper_95": ci["ci_upper_95"],
                "bootstrap_std": ci["bootstrap_std"],
            }
        )

    metrics_path = args.output_dir / "fusion_gate_v2_statistical_metrics_v1.csv"
    ece_bins_path = args.output_dir / "fusion_gate_v2_ece_bins_v1.csv"
    roc_curve_path = args.output_dir / "fusion_gate_v2_low_far_roc_curve_v1.csv"
    summary_path = args.output_dir / "fusion_gate_v2_statistical_summary_v1.json"
    calibration_plot_path = args.output_dir / "fusion_gate_v2_calibration_plot_v1.png"
    low_far_plot_path = args.output_dir / "fusion_gate_v2_low_far_roc_plot_v1.png"

    write_csv(metrics_path, metric_rows)
    write_csv(ece_bins_path, ece_bin_rows)
    write_csv(roc_curve_path, roc_rows)

    create_calibration_plot(
        ece_bin_rows=ece_bin_rows,
        output_path=calibration_plot_path,
    )

    create_low_far_roc_plot(
        roc_rows=roc_rows,
        output_path=low_far_plot_path,
        max_far=0.10,
    )

    summary = {
        "experiment": "fusion_gate_v2_statistical_evaluation_v1",
        "prediction_file": str(args.predictions),
        "threshold": float(args.threshold),
        "known_test_rows": len(known_rows),
        "unknown_test_rows": len(unknown_rows),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_confidence_level": 0.95,
        "n_ece_bins": args.n_bins,
        "point_metrics": point_metrics,
        "confidence_intervals": confidence_intervals,
        "calibration": {
            "brier_score": point_metrics["brier_score"],
            "ece": ece_value,
            "n_bins": args.n_bins,
        },
        "partial_auroc": {
            "standardized_pauc_fpr_0_05": point_metrics["standardized_pauc_fpr_0_05"],
            "standardized_pauc_fpr_0_10": point_metrics["standardized_pauc_fpr_0_10"],
            "note": (
                "pAUC is computed using sklearn's standardized partial AUROC. "
                "The max_fpr value represents the low false-acceptance-rate region."
            ),
        },
        "output_files": {
            "summary": str(summary_path),
            "metrics": str(metrics_path),
            "ece_bins": str(ece_bins_path),
            "low_far_roc_curve": str(roc_curve_path),
            "calibration_plot": str(calibration_plot_path),
            "low_far_roc_plot": str(low_far_plot_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
