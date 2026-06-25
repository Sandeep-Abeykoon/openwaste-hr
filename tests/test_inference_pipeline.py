import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.inference.single_image_inference import (  # noqa: E402
    build_inference_result_payload,
    build_prediction_dataframe,
    extract_state_dict_from_checkpoint,
    normalize_state_dict_keys,
    resolve_class_names,
)


def test_normalize_state_dict_keys_removes_module_prefix():
    state_dict = {
        "module.features.0.weight": torch.zeros(1),
        "module.classifier.weight": torch.ones(1),
    }

    normalized = normalize_state_dict_keys(state_dict)

    assert "features.0.weight" in normalized
    assert "classifier.weight" in normalized


def test_extract_state_dict_from_checkpoint_model_state_dict():
    checkpoint = {
        "model_state_dict": {
            "module.classifier.weight": torch.ones(1),
        }
    }

    state_dict = extract_state_dict_from_checkpoint(checkpoint)

    assert "classifier.weight" in state_dict


def test_resolve_class_names_from_checkpoint_idx_to_class():
    checkpoint = {
        "idx_to_class": {
            1: "plastic",
            0: "paper_cardboard",
        }
    }

    class_names = resolve_class_names(
        checkpoint=checkpoint,
        config_class_names=[],
    )

    assert class_names == ["paper_cardboard", "plastic"]


def test_resolve_class_names_from_config_when_checkpoint_missing_labels():
    class_names = resolve_class_names(
        checkpoint={},
        config_class_names=["paper_cardboard", "plastic"],
    )

    assert class_names == ["paper_cardboard", "plastic"]


def test_build_prediction_dataframe():
    prediction_df = build_prediction_dataframe(
        sample_id="test_001",
        image_path="image.jpg",
        class_names=["paper_cardboard", "plastic", "residual"],
        probabilities=[0.10, 0.80, 0.10],
    )

    assert prediction_df.loc[0, "sample_id"] == "test_001"
    assert prediction_df.loc[0, "pred_label"] == "plastic"
    assert prediction_df.loc[0, "max_softmax_confidence"] == 0.80
    assert "prob_plastic" in prediction_df.columns


def test_build_prediction_dataframe_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        build_prediction_dataframe(
            sample_id="test_001",
            image_path="image.jpg",
            class_names=["plastic"],
            probabilities=[0.80, 0.20],
        )


def test_build_inference_result_payload():
    decisions_df = pd.DataFrame(
        [
            {
                "sample_id": "test_001",
                "image_path": "image.jpg",
                "pred_label": "plastic",
                "max_softmax_confidence": 0.91,
                "top_coarse_label": "recyclable",
                "top_coarse_confidence": 0.95,
                "second_coarse_confidence": 0.05,
                "coarse_margin": 0.90,
                "hierarchical_decision_type": "fine_label",
                "hierarchical_final_label": "plastic",
                "hierarchical_final_confidence": 0.91,
                "hierarchical_decision_reason": "fine_confidence_high",
            }
        ]
    )

    payload = build_inference_result_payload(
        decisions_df=decisions_df,
        class_names=["paper_cardboard", "plastic", "residual"],
        probabilities=[0.04, 0.91, 0.05],
        policy_config={
            "fine_confidence_threshold": 0.90,
            "coarse_confidence_threshold": 0.80,
            "coarse_margin_threshold": 0.65,
            "minimum_confidence_for_coarse": 0.65,
        },
        device=torch.device("cpu"),
    )

    assert payload["hierarchical_decision_type"] == "fine_label"
    assert payload["hierarchical_final_label"] == "plastic"
    assert payload["class_probabilities"]["plastic"] == 0.91