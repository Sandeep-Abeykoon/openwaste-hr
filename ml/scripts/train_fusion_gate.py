from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


KNOWN_CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]


BASE_FEATURES = [
    "confidence",
    "temperature_scaled_confidence",
    "max_logit",
    "energy",
    "softmax_margin",
    "softmax_entropy",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")

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


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("split", "")).strip(),
        str(row.get("image_path", "")).strip(),
        str(row.get("true_label", "")).strip(),
    )


def merge_temperature_confidence(
    *,
    normal_rows: list[dict[str, str]],
    temperature_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    temperature_lookup = {
        row_key(row): row
        for row in temperature_rows
    }

    merged_rows: list[dict[str, str]] = []

    missing_count = 0

    for row in normal_rows:
        merged = dict(row)

        temp_row = temperature_lookup.get(row_key(row))

        if temp_row is None:
            missing_count += 1
            merged["temperature_scaled_confidence"] = merged.get("confidence", "")
        else:
            merged["temperature_scaled_confidence"] = temp_row.get(
                "temperature_scaled_confidence",
                merged.get("confidence", ""),
            )

        merged_rows.append(merged)

    if missing_count:
        print(
            f"Warning: {missing_count} rows did not match temperature-scaled predictions. "
            "Raw confidence was used as fallback.",
            flush=True,
        )

    return merged_rows


def find_probability_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []

    available_columns = set(rows[0].keys())
    probability_columns: list[str] = []

    for class_name in KNOWN_CLASS_NAMES:
        candidates = [
            f"prob_{class_name}",
            f"probability_{class_name}",
            f"p_{class_name}",
            f"{class_name}_prob",
            f"{class_name}_probability",
        ]

        matched = None

        for candidate in candidates:
            if candidate in available_columns:
                matched = candidate
                break

        if matched is None:
            return []

        probability_columns.append(matched)

    return probability_columns


def probability_vector(
    *,
    row: dict[str, str],
    probability_columns: list[str],
) -> np.ndarray:
    if probability_columns:
        values = np.array(
            [to_float(row.get(column, "0")) for column in probability_columns],
            dtype=float,
        )

        total = float(np.sum(values))

        if total > 0.0:
            return values / total

    confidence = to_float(row.get("confidence", "0"))
    confidence = min(max(confidence, 0.0), 1.0)

    if len(KNOWN_CLASS_NAMES) <= 1:
        return np.array([1.0], dtype=float)

    remainder = max(1.0 - confidence, 0.0)
    other_value = remainder / (len(KNOWN_CLASS_NAMES) - 1)

    values = np.array(
        [confidence] + [other_value] * (len(KNOWN_CLASS_NAMES) - 1),
        dtype=float,
    )

    return values


def add_derived_features(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probability_columns = find_probability_columns(rows)

    feature_rows: list[dict[str, Any]] = []

    for row in rows:
        updated: dict[str, Any] = dict(row)

        probs = probability_vector(
            row=row,
            probability_columns=probability_columns,
        )

        sorted_probs = np.sort(probs)[::-1]

        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0

        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))

        updated["softmax_margin"] = top1 - top2
        updated["softmax_entropy"] = entropy

        feature_rows.append(updated)

    metadata = {
        "probability_columns_detected": probability_columns,
        "probability_columns_available": bool(probability_columns),
        "derived_features": [
            "softmax_margin",
            "softmax_entropy",
        ],
        "note": (
            "If class probability columns were unavailable, softmax margin and entropy "
            "were approximated from top-1 confidence."
        ),
    }

    return feature_rows, metadata


def is_known_split(row: dict[str, Any]) -> bool:
    return str(row.get("split", "")).startswith("known")


def is_unknown_split(row: dict[str, Any]) -> bool:
    return str(row.get("split", "")).startswith("unknown")


def is_correct(row: dict[str, Any]) -> bool:
    return str(row.get("true_label", "")).strip() == str(row.get("pred_label", "")).strip()


def split_rows(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if str(row.get("split", "")).strip() == split_name
    ]


def build_feature_matrix(
    *,
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> np.ndarray:
    matrix = []

    for row in rows:
        matrix.append(
            [to_float(row.get(feature_name, "0")) for feature_name in feature_names]
        )

    return np.array(matrix, dtype=float)


def build_knownness_labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([1 if is_known_split(row) else 0 for row in rows], dtype=int)


def evaluate_threshold(
    *,
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    known_accept = known_scores >= threshold
    unknown_accept = unknown_scores >= threshold

    accepted_known_rows = [
        row for row, accepted in zip(known_rows, known_accept)
        if bool(accepted)
    ]

    known_coverage = float(np.mean(known_accept)) if len(known_accept) else 0.0
    known_rejection_rate = 1.0 - known_coverage

    unknown_acceptance_rate = float(np.mean(unknown_accept)) if len(unknown_accept) else 0.0
    unknown_rejection_rate = 1.0 - unknown_acceptance_rate

    known_accuracy_all = (
        sum(1 for row in known_rows if is_correct(row)) / len(known_rows)
        if known_rows else 0.0
    )

    known_accuracy_accepted = (
        sum(1 for row in accepted_known_rows if is_correct(row)) / len(accepted_known_rows)
        if accepted_known_rows else 0.0
    )

    selective_risk = 1.0 - known_accuracy_accepted if accepted_known_rows else 1.0

    binary_balanced_accuracy = (known_coverage + unknown_rejection_rate) / 2.0

    false_acceptance_rate = unknown_acceptance_rate
    false_rejection_rate = known_rejection_rate

    return {
        "threshold": float(threshold),
        "known_rows": len(known_rows),
        "unknown_rows": len(unknown_rows),
        "known_coverage": float(known_coverage),
        "known_rejection_rate": float(known_rejection_rate),
        "unknown_rejection_rate": float(unknown_rejection_rate),
        "unknown_acceptance_rate": float(unknown_acceptance_rate),
        "false_acceptance_rate": float(false_acceptance_rate),
        "false_rejection_rate": float(false_rejection_rate),
        "known_accuracy_all": float(known_accuracy_all),
        "known_accuracy_accepted": float(known_accuracy_accepted),
        "selective_risk": float(selective_risk),
        "binary_balanced_accuracy": float(binary_balanced_accuracy),
    }


def select_threshold(
    *,
    known_rows: list[dict[str, Any]],
    unknown_rows: list[dict[str, Any]],
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_known_coverage_floor: float,
) -> dict[str, Any]:
    all_scores = np.concatenate([known_scores, unknown_scores])
    thresholds = np.unique(all_scores)

    if len(thresholds) > 1000:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, 1000))

    candidates: list[dict[str, Any]] = []

    for threshold in thresholds:
        result = evaluate_threshold(
            known_rows=known_rows,
            unknown_rows=unknown_rows,
            known_scores=known_scores,
            unknown_scores=unknown_scores,
            threshold=float(threshold),
        )

        candidates.append(result)

    feasible = [
        result for result in candidates
        if result["known_coverage"] >= target_known_coverage_floor
    ]

    if feasible:
        selected = max(
            feasible,
            key=lambda result: (
                result["unknown_rejection_rate"],
                result["known_accuracy_accepted"],
                result["binary_balanced_accuracy"],
                result["known_coverage"],
            ),
        )

        selected["threshold_selection_mode"] = (
            "minimize_false_acceptance_subject_to_known_coverage_floor"
        )

        return selected

    selected = max(
        candidates,
        key=lambda result: (
            result["binary_balanced_accuracy"],
            result["known_accuracy_accepted"],
            result["unknown_rejection_rate"],
            result["known_coverage"],
        ),
    )

    selected["threshold_selection_mode"] = "fallback_best_binary_balanced_accuracy"

    return selected


def add_fusion_predictions(
    *,
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []

    for row, score in zip(rows, scores):
        updated = dict(row)

        accepted = bool(score >= threshold)

        updated["fusion_knownness_score"] = float(score)
        updated["fusion_threshold"] = float(threshold)
        updated["fusion_accepted_as_known"] = accepted
        updated["fusion_decision"] = (
            "known_fine_label" if accepted else "unknown_manual_review"
        )

        if accepted:
            updated["fusion_user_visible_label"] = updated.get("pred_label", "")
            updated["fusion_reject_reason"] = ""
        else:
            updated["fusion_user_visible_label"] = ""
            updated["fusion_reject_reason"] = "fusion_knownness_below_threshold"

        output_rows.append(updated)

    return output_rows


def tune_logistic_regression_c(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_values: list[float],
) -> dict[str, Any]:
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    tuning_rows: list[dict[str, Any]] = []

    best_row: dict[str, Any] | None = None

    for c_value in c_values:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "gate",
                    LogisticRegression(
                        penalty="l2",
                        C=c_value,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )

        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            scoring="roc_auc",
            cv=cv,
        )

        row = {
            "C": float(c_value),
            "cv_auroc_mean": float(np.mean(scores)),
            "cv_auroc_std": float(np.std(scores)),
        }

        tuning_rows.append(row)

        if best_row is None or row["cv_auroc_mean"] > best_row["cv_auroc_mean"]:
            best_row = row

    if best_row is None:
        raise ValueError("Could not tune Logistic Regression C.")

    return {
        "best_C": best_row["C"],
        "rows": tuning_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate calibrated multi-score fusion gate for OpenWaste-HR."
    )

    parser.add_argument(
        "--open-set-dir",
        type=Path,
        default=Path("ml/outputs/reject_option/stage_04_final_open_set_v1"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/outputs/fusion_gate/stage_04_fusion_gate_v1"),
    )

    parser.add_argument(
        "--target-known-coverage-floor",
        type=float,
        default=0.75,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    normal_predictions_path = args.open_set_dir / "open_set_predictions_v1.csv"
    temperature_predictions_path = args.open_set_dir / "temperature_scaled_predictions_v1.csv"

    normal_rows = read_csv(normal_predictions_path)
    temperature_rows = read_csv(temperature_predictions_path)

    merged_rows = merge_temperature_confidence(
        normal_rows=normal_rows,
        temperature_rows=temperature_rows,
    )

    feature_rows, feature_metadata = add_derived_features(merged_rows)

    known_val_rows = split_rows(feature_rows, "known_val")
    unknown_val_rows = split_rows(feature_rows, "unknown_val")
    known_test_rows = split_rows(feature_rows, "known_test")
    unknown_test_rows = split_rows(feature_rows, "unknown_test")

    calibration_rows = known_val_rows + unknown_val_rows
    test_rows = known_test_rows + unknown_test_rows

    x_calibration = build_feature_matrix(
        rows=calibration_rows,
        feature_names=BASE_FEATURES,
    )

    y_calibration = build_knownness_labels(calibration_rows)

    tuning = tune_logistic_regression_c(
        x_train=x_calibration,
        y_train=y_calibration,
        c_values=[0.1, 1.0, 3.0, 10.0],
    )

    best_c = float(tuning["best_C"])

    fusion_gate = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "gate",
                LogisticRegression(
                    penalty="l2",
                    C=best_c,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    fusion_gate.fit(x_calibration, y_calibration)

    x_known_val = build_feature_matrix(rows=known_val_rows, feature_names=BASE_FEATURES)
    x_unknown_val = build_feature_matrix(rows=unknown_val_rows, feature_names=BASE_FEATURES)
    x_known_test = build_feature_matrix(rows=known_test_rows, feature_names=BASE_FEATURES)
    x_unknown_test = build_feature_matrix(rows=unknown_test_rows, feature_names=BASE_FEATURES)

    known_val_scores = fusion_gate.predict_proba(x_known_val)[:, 1]
    unknown_val_scores = fusion_gate.predict_proba(x_unknown_val)[:, 1]
    known_test_scores = fusion_gate.predict_proba(x_known_test)[:, 1]
    unknown_test_scores = fusion_gate.predict_proba(x_unknown_test)[:, 1]

    selected_val_threshold = select_threshold(
        known_rows=known_val_rows,
        unknown_rows=unknown_val_rows,
        known_scores=known_val_scores,
        unknown_scores=unknown_val_scores,
        target_known_coverage_floor=args.target_known_coverage_floor,
    )

    threshold = float(selected_val_threshold["threshold"])

    val_result = evaluate_threshold(
        known_rows=known_val_rows,
        unknown_rows=unknown_val_rows,
        known_scores=known_val_scores,
        unknown_scores=unknown_val_scores,
        threshold=threshold,
    )

    test_result = evaluate_threshold(
        known_rows=known_test_rows,
        unknown_rows=unknown_test_rows,
        known_scores=known_test_scores,
        unknown_scores=unknown_test_scores,
        threshold=threshold,
    )

    val_auroc = float(
        roc_auc_score(
            [1] * len(known_val_scores) + [0] * len(unknown_val_scores),
            np.concatenate([known_val_scores, unknown_val_scores]),
        )
    )

    test_auroc = float(
        roc_auc_score(
            [1] * len(known_test_scores) + [0] * len(unknown_test_scores),
            np.concatenate([known_test_scores, unknown_test_scores]),
        )
    )

    val_result["auroc_known_vs_unknown"] = val_auroc
    test_result["auroc_known_vs_unknown"] = test_auroc

    x_all = build_feature_matrix(rows=feature_rows, feature_names=BASE_FEATURES)
    all_scores = fusion_gate.predict_proba(x_all)[:, 1]

    fusion_prediction_rows = add_fusion_predictions(
        rows=feature_rows,
        scores=all_scores,
        threshold=threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "fusion_gate_model_v1.joblib"
    joblib.dump(fusion_gate, model_path)

    write_csv(args.output_dir / "fusion_gate_predictions_v1.csv", fusion_prediction_rows)
    write_csv(args.output_dir / "fusion_gate_c_tuning_v1.csv", tuning["rows"])

    metric_rows = [
        {
            "method": "fusion_gate_v1",
            "split": "validation",
            **val_result,
        },
        {
            "method": "fusion_gate_v1",
            "split": "test",
            **test_result,
        },
    ]

    write_csv(args.output_dir / "fusion_gate_metrics_v1.csv", metric_rows)

    summary = {
        "experiment": "stage_04_fusion_gate_v1",
        "base_model": "stage_04_add_trashbox_clean_v1",
        "fusion_model": "LogisticRegression_L2_with_StandardScaler",
        "feature_names": BASE_FEATURES,
        "feature_metadata": feature_metadata,
        "training_rows": {
            "known_val": len(known_val_rows),
            "unknown_val": len(unknown_val_rows),
            "known_test": len(known_test_rows),
            "unknown_test": len(unknown_test_rows),
        },
        "target_known_coverage_floor": args.target_known_coverage_floor,
        "c_tuning": tuning,
        "selected_threshold_from_validation": selected_val_threshold,
        "validation_result": val_result,
        "test_result": test_result,
        "model_path": str(model_path),
        "output_files": {
            "summary": str(args.output_dir / "fusion_gate_summary_v1.json"),
            "metrics": str(args.output_dir / "fusion_gate_metrics_v1.csv"),
            "predictions": str(args.output_dir / "fusion_gate_predictions_v1.csv"),
            "c_tuning": str(args.output_dir / "fusion_gate_c_tuning_v1.csv"),
            "model": str(model_path),
        },
        "rule": (
            "Fusion gate is trained using known_val + unknown_val only. "
            "The accept/reject threshold is selected on validation only. "
            "Final reporting uses known_test + unknown_test."
        ),
    }

    (args.output_dir / "fusion_gate_summary_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    policy = {
        "policy": {
            "name": "fusion_gate_policy_v1",
            "base_model": "stage_04_add_trashbox_clean_v1",
            "fusion_gate_model_path": str(model_path),
            "threshold": threshold,
            "decision_rule": "accept_known_if_fusion_knownness_score_greater_or_equal_threshold",
            "feature_names": BASE_FEATURES,
            "target_known_coverage_floor": args.target_known_coverage_floor,
        },
        "test_reference": {
            "known_coverage": test_result["known_coverage"],
            "unknown_rejection_rate": test_result["unknown_rejection_rate"],
            "false_acceptance_rate": test_result["false_acceptance_rate"],
            "false_rejection_rate": test_result["false_rejection_rate"],
            "known_accuracy_accepted": test_result["known_accuracy_accepted"],
            "auroc_known_vs_unknown": test_auroc,
        },
    }

    (args.output_dir / "fusion_gate_policy_v1.yaml").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
