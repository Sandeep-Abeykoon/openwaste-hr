import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.training.image_transforms import build_eval_transform  # noqa: E402
from openwaste_hr.training.torch_dataset import TorchManifestImageDataset  # noqa: E402


def create_test_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size)
    image.save(path)


def create_manifest(project_root: Path) -> Path:
    image_1 = project_root / "ml" / "data" / "raw" / "test" / "plastic_001.jpg"
    image_2 = project_root / "ml" / "data" / "raw" / "test" / "metal_001.jpg"

    create_test_image(image_1)
    create_test_image(image_2)

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "source_dataset": "test",
                "source_split": "train",
                "image_path": "ml/data/raw/test/plastic_001.jpg",
                "original_label": "plastic",
                "fine_label": "plastic",
                "coarse_label": "recyclable",
                "is_known": "true",
                "usage": "known_train",
                "license_notes": "test",
            },
            {
                "sample_id": "sample_002",
                "source_dataset": "test",
                "source_split": "val",
                "image_path": "ml/data/raw/test/metal_001.jpg",
                "original_label": "metal",
                "fine_label": "metal",
                "coarse_label": "recyclable",
                "is_known": "true",
                "usage": "known_val",
                "license_notes": "test",
            },
        ]
    )

    manifest_path = project_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    return manifest_path


def test_torch_manifest_dataset_returns_tensor_sample(tmp_path):
    manifest_path = create_manifest(tmp_path)

    dataset = TorchManifestImageDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
        label_to_id={"plastic": 0, "metal": 1},
        usage_filter=["known_train", "known_val"],
        label_column="fine_label",
        transform=build_eval_transform(image_size=32),
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["image"].shape == (3, 32, 32)
    assert sample["label"].item() in {0, 1}
    assert sample["fine_label"] in {"plastic", "metal"}


def test_torch_manifest_dataset_rejects_missing_label_mapping(tmp_path):
    manifest_path = create_manifest(tmp_path)

    with pytest.raises(ValueError, match="labels missing from label_to_id"):
        TorchManifestImageDataset(
            manifest_path=manifest_path,
            project_root=tmp_path,
            label_to_id={"plastic": 0},
            usage_filter=["known_train", "known_val"],
            label_column="fine_label",
            transform=build_eval_transform(image_size=32),
        )