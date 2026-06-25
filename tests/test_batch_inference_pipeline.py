import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.inference.batch_inference import (  # noqa: E402
    build_batch_summary,
    build_decision_distribution,
    build_final_label_distribution,
    list_image_files,
    make_project_relative_path,
)


def create_decisions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "local_000001",
                "hierarchical_decision_type": "fine_label",
                "hierarchical_final_label": "plastic",
            },
            {
                "sample_id": "local_000002",
                "hierarchical_decision_type": "coarse_label",
                "hierarchical_final_label": "recyclable",
            },
            {
                "sample_id": "local_000003",
                "hierarchical_decision_type": "manual_review",
                "hierarchical_final_label": "manual_review",
            },
        ]
    )


def test_list_image_files_sorted_and_filtered(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    (image_dir / "b.jpg").write_text("fake", encoding="utf-8")
    (image_dir / "a.png").write_text("fake", encoding="utf-8")
    (image_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    image_files = list_image_files(
        image_dir=image_dir,
        image_globs=["*.jpg", "*.png"],
    )

    assert [path.name for path in image_files] == ["a.png", "b.jpg"]


def test_list_image_files_respects_max_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    (image_dir / "a.jpg").write_text("fake", encoding="utf-8")
    (image_dir / "b.jpg").write_text("fake", encoding="utf-8")
    (image_dir / "c.jpg").write_text("fake", encoding="utf-8")

    image_files = list_image_files(
        image_dir=image_dir,
        image_globs=["*.jpg"],
        max_images=2,
    )

    assert len(image_files) == 2
    assert [path.name for path in image_files] == ["a.jpg", "b.jpg"]


def test_list_image_files_raises_for_empty_directory(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    with pytest.raises(ValueError, match="No image files"):
        list_image_files(
            image_dir=image_dir,
            image_globs=["*.jpg"],
        )


def test_make_project_relative_path(tmp_path):
    project_root = tmp_path / "project"
    image_path = project_root / "ml" / "data" / "image.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("fake", encoding="utf-8")

    relative_path = make_project_relative_path(
        project_root=project_root,
        file_path=image_path,
    )

    assert relative_path == "ml/data/image.jpg"


def test_build_batch_summary():
    decisions_df = create_decisions_df()

    summary = build_batch_summary(decisions_df)

    assert summary["total_images"] == 3
    assert summary["fine_label_count"] == 1
    assert summary["coarse_label_count"] == 1
    assert summary["manual_review_count"] == 1
    assert summary["accepted_count"] == 2


def test_build_decision_distribution():
    decisions_df = create_decisions_df()

    distribution_df = build_decision_distribution(decisions_df)

    assert set(distribution_df["decision_type"]) == {
        "fine_label",
        "coarse_label",
        "manual_review",
    }
    assert int(distribution_df["count"].sum()) == 3


def test_build_final_label_distribution():
    decisions_df = create_decisions_df()

    distribution_df = build_final_label_distribution(decisions_df)

    assert set(distribution_df["final_label"]) == {
        "plastic",
        "recyclable",
        "manual_review",
    }
    assert int(distribution_df["count"].sum()) == 3