import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.data.inspect_dataset import (  # noqa: E402
    build_count_table,
    build_dimension_summary,
    fail_if_invalid_images,
    inspect_image_files,
)


def create_test_image(path: Path, size: tuple[int, int] = (40, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size)
    image.save(path)


def test_inspect_image_files_reads_valid_images(tmp_path):
    image_path = tmp_path / "ml" / "data" / "raw" / "test" / "img001.jpg"
    create_test_image(image_path, size=(40, 30))

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "image_path": "ml/data/raw/test/img001.jpg",
            }
        ]
    )

    report = inspect_image_files(manifest, tmp_path)

    assert len(report) == 1
    assert report.loc[0, "exists"] is True or report.loc[0, "exists"] == True
    assert report.loc[0, "readable"] is True or report.loc[0, "readable"] == True
    assert report.loc[0, "width"] == 40
    assert report.loc[0, "height"] == 30


def test_fail_if_invalid_images_raises_for_missing_image(tmp_path):
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "image_path": "ml/data/raw/test/missing.jpg",
            }
        ]
    )

    report = inspect_image_files(manifest, tmp_path)

    with pytest.raises(ValueError, match="Missing or unreadable images found"):
        fail_if_invalid_images(report)


def test_build_count_table_and_dimension_summary(tmp_path):
    image_path_1 = tmp_path / "ml" / "data" / "raw" / "test" / "img001.jpg"
    image_path_2 = tmp_path / "ml" / "data" / "raw" / "test" / "img002.jpg"

    create_test_image(image_path_1, size=(40, 30))
    create_test_image(image_path_2, size=(20, 10))

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "image_path": "ml/data/raw/test/img001.jpg",
                "fine_label": "plastic",
            },
            {
                "sample_id": "sample_002",
                "image_path": "ml/data/raw/test/img002.jpg",
                "fine_label": "plastic",
            },
        ]
    )

    count_table = build_count_table(manifest, "fine_label")
    image_report = inspect_image_files(manifest, tmp_path)
    dimension_summary = build_dimension_summary(image_report)

    assert count_table.loc[0, "fine_label"] == "plastic"
    assert count_table.loc[0, "count"] == 2
    assert "mean_width" in set(dimension_summary["metric"])
    assert "mean_height" in set(dimension_summary["metric"])