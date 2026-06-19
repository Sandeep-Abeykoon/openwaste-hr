import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.evaluation.classification_metrics import (  # noqa: E402
    build_classification_report_dataframe,
    build_confusion_matrix_dataframe,
    build_prediction_dataframe,
    compute_summary_metrics,
)


def test_compute_summary_metrics():
    y_true = [0, 1, 1, 2]
    y_pred = [0, 1, 2, 2]
    label_names = ["plastic", "metal", "residual"]

    metrics = compute_summary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    assert round(metrics["accuracy"], 2) == 0.75
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "balanced_accuracy" in metrics


def test_confusion_matrix_and_classification_report_shapes():
    y_true = [0, 1, 1, 2]
    y_pred = [0, 1, 2, 2]
    label_names = ["plastic", "metal", "residual"]

    confusion_df = build_confusion_matrix_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    report_df = build_classification_report_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        label_names=label_names,
    )

    assert confusion_df.shape == (3, 3)
    assert "true_plastic" in confusion_df.index
    assert "pred_residual" in confusion_df.columns
    assert "plastic" in set(report_df["label"])
    assert "macro avg" in set(report_df["label"])


def test_build_prediction_dataframe():
    metadata_rows = [
        {
            "sample_id": "sample_001",
            "image_path": "image.jpg",
            "source_dataset": "test",
            "source_split": "test",
            "original_label": "plastic",
            "fine_label": "plastic",
            "coarse_label": "recyclable",
            "usage": "known_test",
        }
    ]

    prediction_df = build_prediction_dataframe(
        metadata_rows=metadata_rows,
        y_true=[0],
        y_pred=[1],
        confidences=[0.82],
        probability_rows=[[0.18, 0.82]],
        label_names=["plastic", "metal"],
    )

    assert prediction_df.loc[0, "true_label"] == "plastic"
    assert prediction_df.loc[0, "pred_label"] == "metal"
    assert prediction_df.loc[0, "correct"] is False or prediction_df.loc[0, "correct"] == False
    assert prediction_df.loc[0, "prob_plastic"] == 0.18
    assert prediction_df.loc[0, "prob_metal"] == 0.82