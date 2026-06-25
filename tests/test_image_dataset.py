import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.data.image_dataset import ManifestImageDataset  # noqa: E402


def create_test_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size)
    image.save(path)


def create_test_manifest(project_root: Path) -> Path:
    image_1 = project_root / "ml" / "data" / "raw" / "test" / "plastic_001.jpg"
    image_2 = project_root / "ml" / "data" / "raw" / "test" / "metal_001.jpg"

    create_test_image(image_1)
    create_test_image(image_2)

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "source_dataset": "test_source",
                "source_split": "train",
                "image_path": "ml/data/raw/test/plastic_001.jpg",
                "original_label": "plastic",
                "fine_label": "plastic",
                "coarse_label": "recyclable",
                "is_known": "true",
                "usage": "known_train",
                "license_notes": "test image",
            },
            {
                "sample_id": "sample_002",
                "source_dataset": "test_source",
                "source_split": "val",
                "image_path": "ml/data/raw/test/metal_001.jpg",
                "original_label": "metal",
                "fine_label": "metal",
                "coarse_label": "recyclable",
                "is_known": "true",
                "usage": "known_val",
                "license_notes": "test image",
            },
        ]
    )

    manifest_path = project_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return manifest_path


def test_manifest_image_dataset_loads_images(tmp_path):
    manifest_path = create_test_manifest(tmp_path)

    dataset = ManifestImageDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    assert len(dataset) == 2

    sample = dataset[0]

    assert sample["image"].mode == "RGB"
    assert sample["sample_id"] == "sample_001"
    assert sample["fine_label"] == "plastic"
    assert sample["coarse_label"] == "recyclable"


def test_manifest_image_dataset_usage_filter(tmp_path):
    manifest_path = create_test_manifest(tmp_path)

    dataset = ManifestImageDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
        usage_filter=["known_val"],
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "sample_002"


def test_manifest_image_dataset_raises_for_missing_image(tmp_path):
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "source_dataset": "test_source",
                "source_split": "train",
                "image_path": "ml/data/raw/test/missing.jpg",
                "original_label": "plastic",
                "fine_label": "plastic",
                "coarse_label": "recyclable",
                "is_known": "true",
                "usage": "known_train",
                "license_notes": "test image",
            }
        ]
    )

    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    dataset = ManifestImageDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        dataset[0]