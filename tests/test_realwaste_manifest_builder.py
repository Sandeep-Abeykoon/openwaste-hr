from pathlib import Path

import pandas as pd
import yaml

from openwaste_hr.data.build_realwaste_manifest import (
    build_realwaste_manifest,
    normalise_label,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_toy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image content")


def test_realwaste_config_exists():
    assert (PROJECT_ROOT / "ml" / "configs" / "realwaste.yaml").exists()
    assert (PROJECT_ROOT / "ml" / "configs" / "realwaste_label_mapping_v1.csv").exists()


def test_realwaste_config_contains_expected_labels():
    config = yaml.safe_load(
        (PROJECT_ROOT / "ml" / "configs" / "realwaste.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert "Cardboard" in config["expected_labels"]
    assert "Food Organics" in config["expected_labels"]
    assert "Textile Trash" in config["expected_labels"]
    assert config["source_dataset"]["name"] == "realwaste"


def test_realwaste_label_mapping_contains_known_and_unknown_roles():
    mapping = pd.read_csv(PROJECT_ROOT / "ml" / "configs" / "realwaste_label_mapping_v1.csv")

    textile_row = mapping.loc[mapping["original_label"] == "Textile Trash"].iloc[0]
    cardboard_row = mapping.loc[mapping["original_label"] == "Cardboard"].iloc[0]

    assert cardboard_row["fine_label"] == "paper_cardboard"
    assert cardboard_row["coarse_label"] == "recyclable"
    assert bool(cardboard_row["is_known"]) is True

    assert textile_row["fine_label"] == "unknown"
    assert textile_row["coarse_label"] == "unknown"
    assert bool(textile_row["is_known"]) is False
    assert textile_row["mapping_role"] == "future_class_candidate"


def test_normalise_label_handles_realwaste_labels():
    assert normalise_label("Food Organics") == "food_organics"
    assert normalise_label("Textile Trash") == "textile_trash"
    assert normalise_label("Miscellaneous Trash") == "miscellaneous_trash"


def test_realwaste_manifest_builder_with_toy_dataset(tmp_path: Path):
    project_root = tmp_path

    raw_root = project_root / "ml" / "data" / "raw" / "realwaste"
    mapping_path = project_root / "ml" / "configs" / "realwaste_label_mapping_v1.csv"
    config_path = project_root / "ml" / "configs" / "realwaste.yaml"

    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    original_mapping = PROJECT_ROOT / "ml" / "configs" / "realwaste_label_mapping_v1.csv"
    mapping_path.write_text(original_mapping.read_text(encoding="utf-8"), encoding="utf-8")

    config = {
        "project": {
            "name": "OpenWaste-HR",
            "stage": "realwaste_intake_test",
            "version": "v1",
        },
        "paths": {
            "raw_root": "ml/data/raw/realwaste",
            "label_mapping_csv": "ml/configs/realwaste_label_mapping_v1.csv",
            "output_dir": "ml/data/splits",
        },
        "outputs": {
            "manifest_csv": "realwaste_manifest_v1.csv",
            "known_train_csv": "realwaste_known_train_v1.csv",
            "known_val_csv": "realwaste_known_val_v1.csv",
            "known_test_csv": "realwaste_known_test_v1.csv",
            "unknown_test_csv": "realwaste_unknown_test_v1.csv",
            "summary_json": "realwaste_manifest_summary_v1.json",
        },
        "split": {
            "random_seed": 42,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "image_extensions": [".jpg", ".png"],
        "source_dataset": {
            "name": "realwaste",
            "license_notes": "CC BY-NC-SA 4.0",
        },
    }

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    for label in ["Cardboard", "Plastic", "Textile Trash"]:
        for index in range(3):
            create_toy_image(raw_root / label / f"sample_{index}.jpg")

    result = build_realwaste_manifest(config_path=config_path, project_root=project_root)

    assert result["total_samples"] == 9
    assert result["known_samples"] == 6
    assert result["unknown_samples"] == 3

    manifest_path = project_root / "ml" / "data" / "splits" / "realwaste_manifest_v1.csv"
    unknown_path = project_root / "ml" / "data" / "splits" / "realwaste_unknown_test_v1.csv"

    manifest = pd.read_csv(manifest_path)
    unknown = pd.read_csv(unknown_path)

    assert set(manifest["source_dataset"]) == {"realwaste"}
    assert "paper_cardboard" in set(manifest["fine_label"])
    assert "plastic" in set(manifest["fine_label"])
    assert set(unknown["fine_label"]) == {"unknown"}
    assert set(unknown["mapping_role"]) == {"future_class_candidate"}
