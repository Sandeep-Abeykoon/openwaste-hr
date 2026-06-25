import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.data.build_local_unknown_manifest import (  # noqa: E402
    build_local_unknown_manifest,
    build_local_unknown_manifest_dataframe,
    validate_collection_sheet,
)


def create_test_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size)
    image.save(path)


def create_collection_sheet(project_root: Path) -> Path:
    image_1 = project_root / "ml" / "data" / "local_unknown" / "images" / "local_000001.jpg"
    image_2 = project_root / "ml" / "data" / "local_unknown" / "images" / "local_000002.jpg"

    create_test_image(image_1)
    create_test_image(image_2)

    collection_df = pd.DataFrame(
        [
            {
                "sample_id": "local_000001",
                "image_path": "ml/data/local_unknown/images/local_000001.jpg",
                "capture_date": "2026-06-19",
                "location_context": "home",
                "object_description": "shiny food wrapper",
                "why_unknown_or_difficult": "Reflective mixed-material packaging",
                "lighting_condition": "indoor",
                "background_condition": "plain",
                "human_note": "manual review candidate",
                "usage": "unknown_test",
            },
            {
                "sample_id": "local_000002",
                "image_path": "ml/data/local_unknown/images/local_000002.jpg",
                "capture_date": "2026-06-19",
                "location_context": "home",
                "object_description": "dirty plastic container",
                "why_unknown_or_difficult": "Contaminated object may confuse classifier",
                "lighting_condition": "indoor",
                "background_condition": "cluttered",
                "human_note": "difficult local sample",
                "usage": "active_learning_candidate",
            },
        ]
    )

    collection_path = project_root / "ml" / "data" / "splits" / "local_unknown_collection_sheet_v1.csv"
    collection_path.parent.mkdir(parents=True, exist_ok=True)
    collection_df.to_csv(collection_path, index=False)

    return collection_path


def create_config(project_root: Path) -> Path:
    taxonomy_source = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"
    taxonomy_target = project_root / "ml" / "configs" / "taxonomy.yaml"
    taxonomy_target.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_target.write_text(taxonomy_source.read_text(encoding="utf-8"), encoding="utf-8")

    config = {
        "project": {
            "name": "OpenWaste-HR",
            "dataset_stage": "local_unknown_manifest",
            "version": "v1",
        },
        "paths": {
            "collection_csv": "ml/data/splits/local_unknown_collection_sheet_v1.csv",
            "output_manifest": "ml/data/splits/local_unknown_manifest_v1.csv",
            "taxonomy": "ml/configs/taxonomy.yaml",
        },
        "rules": {
            "source_dataset": "local_phone_images",
            "source_split": "unknown_test",
            "original_label": "unknown",
            "fine_label": "unknown",
            "coarse_label": "unknown",
            "is_known": False,
            "license_notes": "Own local phone image",
            "allowed_usage": [
                "unknown_test",
                "local_unknown",
                "active_learning_candidate",
            ],
            "require_image_files_exist": True,
        },
    }

    config_path = project_root / "ml" / "configs" / "local_unknown_manifest.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    return config_path


def test_validate_collection_sheet_passes(tmp_path):
    collection_path = create_collection_sheet(tmp_path)
    collection_df = pd.read_csv(collection_path)

    assert validate_collection_sheet(
        collection_df=collection_df,
        project_root=tmp_path,
        allowed_usage=[
            "unknown_test",
            "local_unknown",
            "active_learning_candidate",
        ],
        require_image_files_exist=True,
    ) is True


def test_build_local_unknown_manifest_dataframe():
    collection_df = pd.DataFrame(
        [
            {
                "sample_id": "local_000001",
                "image_path": "ml/data/local_unknown/images/local_000001.jpg",
                "capture_date": "2026-06-19",
                "location_context": "home",
                "object_description": "shiny wrapper",
                "why_unknown_or_difficult": "reflective object",
                "lighting_condition": "indoor",
                "background_condition": "plain",
                "human_note": "test",
                "usage": "unknown_test",
            }
        ]
    )

    rules = {
        "source_dataset": "local_phone_images",
        "original_label": "unknown",
        "fine_label": "unknown",
        "coarse_label": "unknown",
        "is_known": False,
        "license_notes": "Own local phone image",
    }

    manifest = build_local_unknown_manifest_dataframe(
        collection_df=collection_df,
        rules=rules,
    )

    assert manifest.loc[0, "source_dataset"] == "local_phone_images"
    assert manifest.loc[0, "fine_label"] == "unknown"
    assert manifest.loc[0, "coarse_label"] == "unknown"
    assert manifest.loc[0, "is_known"] == "false"
    assert manifest.loc[0, "usage"] == "unknown_test"


def test_build_local_unknown_manifest_saves_file(tmp_path):
    create_collection_sheet(tmp_path)
    config_path = create_config(tmp_path)

    manifest = build_local_unknown_manifest(
        config_path=config_path,
        project_root=tmp_path,
    )

    output_path = tmp_path / "ml" / "data" / "splits" / "local_unknown_manifest_v1.csv"

    assert len(manifest) == 2
    assert output_path.exists()
    assert set(manifest["fine_label"]) == {"unknown"}


def test_validate_collection_sheet_rejects_missing_columns(tmp_path):
    collection_df = pd.DataFrame(
        [
            {
                "sample_id": "local_000001",
                "image_path": "ml/data/local_unknown/images/local_000001.jpg",
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        validate_collection_sheet(
            collection_df=collection_df,
            project_root=tmp_path,
            allowed_usage=["unknown_test"],
            require_image_files_exist=False,
        )


def test_validate_collection_sheet_rejects_invalid_usage(tmp_path):
    collection_path = create_collection_sheet(tmp_path)
    collection_df = pd.read_csv(collection_path)
    collection_df.loc[0, "usage"] = "bad_usage"

    with pytest.raises(ValueError, match="Invalid usage values"):
        validate_collection_sheet(
            collection_df=collection_df,
            project_root=tmp_path,
            allowed_usage=[
                "unknown_test",
                "local_unknown",
                "active_learning_candidate",
            ],
            require_image_files_exist=False,
        )


def test_validate_collection_sheet_rejects_missing_image_when_required(tmp_path):
    collection_df = pd.DataFrame(
        [
            {
                "sample_id": "local_000001",
                "image_path": "ml/data/local_unknown/images/missing.jpg",
                "capture_date": "2026-06-19",
                "location_context": "home",
                "object_description": "missing image",
                "why_unknown_or_difficult": "test",
                "lighting_condition": "indoor",
                "background_condition": "plain",
                "human_note": "test",
                "usage": "unknown_test",
            }
        ]
    )

    with pytest.raises(FileNotFoundError, match="image files do not exist"):
        validate_collection_sheet(
            collection_df=collection_df,
            project_root=tmp_path,
            allowed_usage=["unknown_test"],
            require_image_files_exist=True,
        )