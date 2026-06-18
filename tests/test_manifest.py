import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.data.manifest import validate_manifest  # noqa: E402


def test_valid_manifest_passes():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "source_dataset": "trashnet",
                "source_split": "train",
                "image_path": "ml/data/raw/trashnet/plastic/img001.jpg",
                "original_label": "plastic",
                "fine_label": "plastic",
                "coarse_label": "recyclable",
                "is_known": "true",
                "usage": "known_train",
                "license_notes": "test row",
            },
            {
                "sample_id": "sample_002",
                "source_dataset": "local_phone_images",
                "source_split": "unknown_test",
                "image_path": "ml/data/local_unknown/local001.jpg",
                "original_label": "unknown",
                "fine_label": "unknown",
                "coarse_label": "unknown",
                "is_known": "false",
                "usage": "unknown_test",
                "license_notes": "own image",
            },
        ]
    )

    assert validate_manifest(manifest, taxonomy_path) is True


def test_manifest_rejects_missing_columns():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "source_dataset": "trashnet",
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        validate_manifest(manifest, taxonomy_path)


def test_manifest_rejects_unknown_label_as_known_sample():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "source_dataset": "trashnet",
                "source_split": "train",
                "image_path": "ml/data/raw/trashnet/unknown/img001.jpg",
                "original_label": "unknown",
                "fine_label": "unknown",
                "coarse_label": "unknown",
                "is_known": "true",
                "usage": "known_train",
                "license_notes": "bad test row",
            }
        ]
    )

    with pytest.raises(ValueError, match="known sample has invalid fine label"):
        validate_manifest(manifest, taxonomy_path)


def test_manifest_rejects_wrong_usage_for_known_sample():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "source_dataset": "trashnet",
                "source_split": "train",
                "image_path": "ml/data/raw/trashnet/plastic/img001.jpg",
                "original_label": "plastic",
                "fine_label": "plastic",
                "coarse_label": "recyclable",
                "is_known": "true",
                "usage": "unknown_test",
                "license_notes": "bad usage",
            }
        ]
    )

    with pytest.raises(ValueError, match="known sample has invalid usage"):
        validate_manifest(manifest, taxonomy_path)