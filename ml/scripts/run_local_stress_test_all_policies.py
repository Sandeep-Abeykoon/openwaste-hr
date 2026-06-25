from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infer_with_fusion_gate_v2_policy import (  # noqa: E402
    FUSION_FEATURES,
    build_fusion_feature_vector,
    compute_mahalanobis_knownness,
    compute_model_scores,
    load_classifier,
    preprocess_image,
    read_yaml,
    run_classifier_with_embedding,
)
from train_image_classifier import build_transforms  # noqa: E402


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


POLICY_THRESHOLDS = {
    "confidence_threshold": 0.998321158016373,
    "temperature_scaled_confidence": 0.9405356681418562,
    "max_logit": 7.173780989790249,
    "energy": -7.177558458400871,
    "fusion_gate_v1_score_only": 0.5012177382379158,
    "mahalanobis_only": -1548.8591121639217,
    "fusion_gate_v2_mahalanobis": 0.6314586412215439,
}


FUSION_V1_FEATURES = [
    "confidence",
    "temperature_scaled_confidence",
    "max_logit",
    "energy",
    "softmax_margin",
    "softmax_entropy",
]


FUSION_V2_FEATURES = FUSION_FEATURES


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


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        raise ValueError(f"No supported image files found in: {image_dir}")

    return image_paths


def load_runtime(
    *,
    training_config_path: Path,
    policy_config_path: Path,
    fusion_gate_v1_model_path: Path,
    fusion_gate_v2_model_path: Path,
    mahalanobis_model_path: Path,
) -> dict[str, Any]:
    training_config = read_yaml(training_config_path)
    policy_config = read_yaml(policy_config_path)

    checkpoint_path = Path(policy_config["base_model"]["checkpoint_path"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    if not fusion_gate_v1_model_path.exists():
        raise FileNotFoundError(f"Fusion Gate v1 model not found: {fusion_gate_v1_model_path}")

    if not fusion_gate_v2_model_path.exists():
        raise FileNotFoundError(f"Fusion Gate v2 model not found: {fusion_gate_v2_model_path}")

    if not mahalanobis_model_path.exists():
        raise FileNotFoundError(f"Mahalanobis model not found: {mahalanobis_model_path}")

    temperature = float(policy_config["temperature_scaling"]["temperature"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, class_names = load_classifier(
        training_config=training_config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    transform = build_transforms(
        Path(training_config["preprocessing"]["config_path"]),
        train=False,
    )

    return {
        "training_config": training_config,
        "policy_config": policy_config,
        "device": device,
        "model": model,
        "class_names": class_names,
        "transform": transform,
        "temperature": temperature,
        "fusion_gate_v1": joblib.load(fusion_gate_v1_model_path),
        "fusion_gate_v2": joblib.load(fusion_gate_v2_model_path),
        "mahalanobis_model": joblib.load(mahalanobis_model_path),
    }


def build_feature_vector(
    *,
    scores: dict[str, Any],
    feature_names: list[str],
) -> np.ndarray:
    values = [float(scores[feature_name]) for feature_name in feature_names]
    return np.array(values, dtype=float).reshape(1, -1)


def decide_accept(
    *,
    policy_name: str,
    scores: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[bool, float | str, float | str]:
    if policy_name == "raw_classifier":
        return True, "", ""

    if policy_name == "confidence_threshold":
        threshold = POLICY_THRESHOLDS["confidence_threshold"]
        return bool(scores["confidence"] >= threshold), float(scores["confidence"]), threshold

    if policy_name == "temperature_scaled_confidence":
        threshold = POLICY_THRESHOLDS["temperature_scaled_confidence"]
        return (
            bool(scores["temperature_scaled_confidence"] >= threshold),
            float(scores["temperature_scaled_confidence"]),
            threshold,
        )

    if policy_name == "max_logit":
        threshold = POLICY_THRESHOLDS["max_logit"]
        return bool(scores["max_logit"] >= threshold), float(scores["max_logit"]), threshold

    if policy_name == "energy":
        threshold = POLICY_THRESHOLDS["energy"]
        return bool(scores["energy"] <= threshold), float(scores["energy"]), threshold

    if policy_name == "fusion_gate_v1_score_only":
        threshold = POLICY_THRESHOLDS["fusion_gate_v1_score_only"]
        vector = build_feature_vector(
            scores=scores,
            feature_names=FUSION_V1_FEATURES,
        )
        knownness_score = float(runtime["fusion_gate_v1"].predict_proba(vector)[0, 1])
        return bool(knownness_score >= threshold), knownness_score, threshold

    if policy_name == "mahalanobis_only":
        threshold = POLICY_THRESHOLDS["mahalanobis_only"]
        return (
            bool(scores["mahalanobis_knownness"] >= threshold),
            float(scores["mahalanobis_knownness"]),
            threshold,
        )

    if policy_name == "fusion_gate_v2_mahalanobis":
        threshold = POLICY_THRESHOLDS["fusion_gate_v2_mahalanobis"]
        vector = build_feature_vector(
            scores=scores,
            feature_names=FUSION_V2_FEATURES,
        )
        knownness_score = float(runtime["fusion_gate_v2"].predict_proba(vector)[0, 1])
        return bool(knownness_score >= threshold), knownness_score, threshold

    raise ValueError(f"Unsupported policy: {policy_name}")


def predict_base_scores(
    *,
    image_path: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    image_tensor = preprocess_image(
        image_path=image_path,
        transform=runtime["transform"],
        device=runtime["device"],
    )

    logits, embedding, embedding_layer = run_classifier_with_embedding(
        model=runtime["model"],
        image_tensor=image_tensor,
    )

    model_scores = compute_model_scores(
        logits=logits,
        class_names=runtime["class_names"],
        temperature=runtime["temperature"],
    )

    mahalanobis_scores = compute_mahalanobis_knownness(
        embedding=embedding,
        mahalanobis_model=runtime["mahalanobis_model"],
        class_names=runtime["class_names"],
    )

    scores = {
        **model_scores,
        **mahalanobis_scores,
    }

    scores["embedding_layer"] = embedding_layer
    scores["embedding_dimension"] = int(embedding.shape[0])

    return scores


def create_prediction_rows_for_image(
    *,
    image_path: Path,
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    scores = predict_base_scores(
        image_path=image_path,
        runtime=runtime,
    )

    policy_names = [
        "raw_classifier",
        "confidence_threshold",
        "temperature_scaled_confidence",
        "max_logit",
        "energy",
        "fusion_gate_v1_score_only",
        "mahalanobis_only",
        "fusion_gate_v2_mahalanobis",
    ]

    rows: list[dict[str, Any]] = []

    for policy_name in policy_names:
        accepted, policy_score, threshold = decide_accept(
            policy_name=policy_name,
            scores=scores,
            runtime=runtime,
        )

        decision_type = "known_fine_label" if accepted else "unknown_manual_review"
        user_visible_label = scores["pred_label"] if accepted else "manual_review_required"
        coarse_label = "recyclable" if accepted else "manual_review_required"

        rows.append(
            {
                "sample_id": image_path.stem,
                "image_path": str(image_path),
                "policy_name": policy_name,
                "stress_test_expected_status": "outside_or_uncertain_local_item",
                "internal_top1_prediction": scores["pred_label"],
                "policy_score": policy_score,
                "policy_threshold": threshold,
                "accepted_as_known": accepted,
                "decision_type": decision_type,
                "user_visible_label": user_visible_label,
                "coarse_label": coarse_label,
                "show_internal_prediction_to_user": accepted,
                "raw_confidence": scores["confidence"],
                "temperature_scaled_confidence": scores["temperature_scaled_confidence"],
                "max_logit": scores["max_logit"],
                "energy": scores["energy"],
                "softmax_margin": scores["softmax_margin"],
                "softmax_entropy": scores["softmax_entropy"],
                "mahalanobis_min_distance": scores["mahalanobis_min_distance"],
                "mahalanobis_knownness": scores["mahalanobis_knownness"],
                "mahalanobis_nearest_class": scores["mahalanobis_nearest_class"],
                "prob_cardboard": scores.get("prob_cardboard", ""),
                "prob_glass": scores.get("prob_glass", ""),
                "prob_metal": scores.get("prob_metal", ""),
                "prob_paper": scores.get("prob_paper", ""),
                "prob_plastic": scores.get("prob_plastic", ""),
                "embedding_layer": scores["embedding_layer"],
                "embedding_dimension": scores["embedding_dimension"],
            }
        )

    return rows


def summarise_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policy_names = list(dict.fromkeys(row["policy_name"] for row in rows))
    summary_rows: list[dict[str, Any]] = []

    for policy_name in policy_names:
        policy_rows = [row for row in rows if row["policy_name"] == policy_name]

        total = len(policy_rows)
        accepted = sum(1 for row in policy_rows if bool(row["accepted_as_known"]))
        manual_review = total - accepted

        label_counts: dict[str, int] = {}

        for row in policy_rows:
            label = str(row["internal_top1_prediction"])
            label_counts[label] = label_counts.get(label, 0) + 1

        policy_scores = [
            row["policy_score"]
            for row in policy_rows
            if isinstance(row["policy_score"], float)
        ]

        summary_rows.append(
            {
                "policy_name": policy_name,
                "total_images": total,
                "accepted_as_known_count": accepted,
                "manual_review_count": manual_review,
                "accepted_as_known_rate": accepted / total if total else 0.0,
                "manual_review_rate": manual_review / total if total else 0.0,
                "internal_top1_prediction_counts_json": json.dumps(
                    label_counts,
                    ensure_ascii=False,
                ),
                "policy_score_mean": float(np.mean(policy_scores)) if policy_scores else "",
                "policy_score_median": float(np.median(policy_scores)) if policy_scores else "",
                "policy_score_min": float(np.min(policy_scores)) if policy_scores else "",
                "policy_score_max": float(np.max(policy_scores)) if policy_scores else "",
            }
        )

    return summary_rows


def create_summary(
    *,
    prediction_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    image_dir: Path,
) -> dict[str, Any]:
    return {
        "experiment": "local_stress_test_all_policies_v1",
        "image_dir": str(image_dir),
        "description": (
            "Independent local household-image stress test. These images are not used "
            "for model training, threshold tuning, or fusion-gate training."
        ),
        "total_unique_images": len(
            set(str(row["image_path"]) for row in prediction_rows)
        ),
        "policies_compared": [
            row["policy_name"]
            for row in summary_rows
        ],
        "policy_summary": summary_rows,
        "research_interpretation_note": (
            "For local stress images expected to be outside or uncertain relative to "
            "the known taxonomy, a higher manual-review rate is safer. Accepted cases "
            "should be manually inspected before being treated as correct known-class predictions."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all reject/manual-review policies on local stress-test images."
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("ml/data/local_stress_unknown/images"),
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("ml/configs/train_stage_04_add_trashbox_clean.yaml"),
    )

    parser.add_argument(
        "--policy-config",
        type=Path,
        default=Path("ml/configs/final_decision_policy_v2_fusion_gate.yaml"),
    )

    parser.add_argument(
        "--fusion-gate-v1-model",
        type=Path,
        default=Path("ml/outputs/fusion_gate/stage_04_fusion_gate_v1/fusion_gate_model_v1.joblib"),
    )

    parser.add_argument(
        "--fusion-gate-v2-model",
        type=Path,
        default=Path(
            "ml/outputs/fusion_gate/stage_04_fusion_gate_v2_mahalanobis/"
            "fusion_gate_v2_mahalanobis_model.joblib"
        ),
    )

    parser.add_argument(
        "--mahalanobis-model",
        type=Path,
        default=Path("ml/outputs/fusion_gate/stage_04_mahalanobis_v1/mahalanobis_model_v1.joblib"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/outputs/local_stress_test/all_policies_v1"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runtime = load_runtime(
        training_config_path=args.training_config,
        policy_config_path=args.policy_config,
        fusion_gate_v1_model_path=args.fusion_gate_v1_model,
        fusion_gate_v2_model_path=args.fusion_gate_v2_model,
        mahalanobis_model_path=args.mahalanobis_model,
    )

    image_paths = list_images(args.image_dir)

    print(f"Using device: {runtime['device']}")
    print(f"Found local stress-test images: {len(image_paths)}")

    prediction_rows: list[dict[str, Any]] = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Predicting {image_path}")
        prediction_rows.extend(
            create_prediction_rows_for_image(
                image_path=image_path,
                runtime=runtime,
            )
        )

    summary_rows = summarise_policy_rows(prediction_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = args.output_dir / "local_stress_test_all_policy_predictions_v1.csv"
    summary_csv_path = args.output_dir / "local_stress_test_all_policy_summary_v1.csv"
    summary_json_path = args.output_dir / "local_stress_test_all_policy_summary_v1.json"

    write_csv(predictions_path, prediction_rows)
    write_csv(summary_csv_path, summary_rows)

    summary = create_summary(
        prediction_rows=prediction_rows,
        summary_rows=summary_rows,
        image_dir=args.image_dir,
    )

    summary["output_files"] = {
        "predictions": str(predictions_path),
        "summary_csv": str(summary_csv_path),
        "summary_json": str(summary_json_path),
    }

    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
