import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.data.build_trashnet_manifest import (  # noqa: E402
    build_trashnet_manifest,
    discover_trashnet_images,
    split_manifest,
)


TRASHNET_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


def create_fake_trashnet_dataset(root: Path, images_per_class: int = 5) -> None:
    for class_name in TRASHNET_CLASSES:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for index in range(images_per_class):
            image_path = class_dir / f"{class_name}_{index:03d}.jpg"
            image_path.write_bytes(b"fake image content")


def test_discover_trashnet_images_maps_labels_correctly(tmp_path):
    dataset_root = tmp_path / "ml" / "data" / "raw" / "trashnet" / "dataset-resized"
    create_fake_trashnet_dataset(dataset_root, images_per_class=5)

    manifest = discover_trashnet_images(dataset_root, tmp_path)

    assert len(manifest) == 30
    assert set(manifest["original_label"]) == set(TRASHNET_CLASSES)

    cardboard_rows = manifest[manifest["original_label"] == "cardboard"]
    assert set(cardboard_rows["fine_label"]) == {"paper_cardboard"}
    assert set(cardboard_rows["coarse_label"]) == {"recyclable"}

    trash_rows = manifest[manifest["original_label"] == "trash"]
    assert set(trash_rows["fine_label"]) == {"residual"}
    assert set(trash_rows["coarse_label"]) == {"residual"}


def test_split_manifest_creates_train_val_test(tmp_path):
    dataset_root = tmp_path / "ml" / "data" / "raw" / "trashnet" / "dataset-resized"
    create_fake_trashnet_dataset(dataset_root, images_per_class=10)

    manifest = discover_trashnet_images(dataset_root, tmp_path)

    split_df = split_manifest(
        manifest=manifest,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert set(split_df["usage"]) == {"known_train", "known_val", "known_test"}
    assert set(split_df["source_split"]) == {"train", "val", "test"}

    for class_name in TRASHNET_CLASSES:
        class_rows = split_df[split_df["original_label"] == class_name]
        assert "known_train" in set(class_rows["usage"])
        assert "known_val" in set(class_rows["usage"])
        assert "known_test" in set(class_rows["usage"])


def test_build_trashnet_manifest_saves_files(tmp_path):
    dataset_root = tmp_path / "ml" / "data" / "raw" / "trashnet" / "dataset-resized"
    create_fake_trashnet_dataset(dataset_root, images_per_class=5)

    config_path = tmp_path / "ml" / "configs" / "trashnet.yaml"
    taxonomy_source = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"
    taxonomy_target = tmp_path / "ml" / "configs" / "taxonomy.yaml"

    taxonomy_target.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_target.write_text(taxonomy_source.read_text(encoding="utf-8"), encoding="utf-8")

    config = {
        "project": {
            "name": "OpenWaste-HR",
            "dataset_source": "trashnet",
            "intake_version": "v1",
        },
        "paths": {
            "dataset_root": "ml/data/raw/trashnet/dataset-resized",
            "output_manifest": "ml/data/splits/trashnet_manifest_v1.csv",
            "output_train": "ml/data/splits/known_train.csv",
            "output_val": "ml/data/splits/known_val.csv",
            "output_test": "ml/data/splits/known_test.csv",
            "taxonomy": "ml/configs/taxonomy.yaml",
        },
        "split": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "seed": 42,
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    manifest = build_trashnet_manifest(
        config_path=config_path,
        project_root=tmp_path,
    )

    assert len(manifest) == 30
    assert (tmp_path / "ml" / "data" / "splits" / "trashnet_manifest_v1.csv").exists()
    assert (tmp_path / "ml" / "data" / "splits" / "known_train.csv").exists()
    assert (tmp_path / "ml" / "data" / "splits" / "known_val.csv").exists()
    assert (tmp_path / "ml" / "data" / "splits" / "known_test.csv").exists()


def test_missing_trashnet_class_folder_raises_error(tmp_path):
    dataset_root = tmp_path / "ml" / "data" / "raw" / "trashnet" / "dataset-resized"
    dataset_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Expected TrashNet class folder not found"):
        discover_trashnet_images(dataset_root, tmp_path)