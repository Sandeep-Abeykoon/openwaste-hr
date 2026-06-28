import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.training.label_encoding import (  # noqa: E402
    build_label_names_from_manifest,
    build_label_to_id,
    encode_labels,
)


def test_build_label_names_from_manifest_uses_taxonomy_order():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"

    manifest = pd.DataFrame(
        [
            {"fine_label": "metal"},
            {"fine_label": "plastic"},
            {"fine_label": "paper"},
        ]
    )

    label_names = build_label_names_from_manifest(
        manifest=manifest,
        taxonomy_path=taxonomy_path,
        label_column="fine_label",
    )

    assert label_names == ["metal", "paper", "plastic"]


def test_build_label_to_id_and_encode_labels():
    label_names = ["metal", "paper", "plastic"]
    label_to_id = build_label_to_id(label_names)

    assert label_to_id == {
        "metal": 0,
        "paper": 1,
        "plastic": 2,
    }

    encoded = encode_labels(
        labels=["plastic", "metal", "paper"],
        label_to_id=label_to_id,
    )

    assert encoded == [2, 0, 1]


def test_unknown_label_in_known_training_manifest_raises_error():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"

    manifest = pd.DataFrame(
        [
            {"fine_label": "plastic"},
            {"fine_label": "biological"},
        ]
    )

    with pytest.raises(ValueError, match="labels not present in taxonomy"):
        build_label_names_from_manifest(
            manifest=manifest,
            taxonomy_path=taxonomy_path,
            label_column="fine_label",
        )
