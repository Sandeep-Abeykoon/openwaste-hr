from pathlib import Path

import pandas as pd
import yaml

from openwaste_hr.data.build_combined_public_manifests import (
    build_combined_public_manifests,
    normalise_manifest_columns,
    validate_unknown_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_manifest_row(
    *,
    sample_id: str,
    source_dataset: str,
    fine_label: str,
    coarse_label: str,
    is_known: bool,
    usage: str,
    mapping_role: str = "known_train_candidate",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_dataset": source_dataset,
        "source_split": usage,
        "image_path": f"dummy/{sample_id}.jpg",
        "original_label": fine_label,
        "fine_label": fine_label,
        "coarse_label": coarse_label,
        "is_known": is_known,
        "usage": usage,
        "license_notes": "test",
        "mapping_role": mapping_role,
        "mapping_notes": "test",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_combined_public_config_exists():
    assert (
        PROJECT_ROOT / "ml" / "configs" / "combined_public_training_manifests.yaml"
    ).exists()


def test_combined_public_config_contains_expected_inputs_and_outputs():
    config = yaml.safe_load(
        (
            PROJECT_ROOT / "ml" / "configs" / "combined_public_training_manifests.yaml"
        ).read_text(encoding="utf-8")
    )

    assert "trashnet" in config["inputs"]
    assert "realwaste" in config["inputs"]
    assert config["outputs"]["combined_known_train_csv"] == "expanded_public_known_train_v1.csv"
    assert "organic" in config["validation"]["require_known_train_labels"]


def test_normalise_manifest_columns_adds_mapping_columns():
    df = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "source_dataset": "trashnet",
                "source_split": "train",
                "image_path": "a.jpg",
                "original_label": "paper",
                "fine_label": "paper_cardboard",
                "coarse_label": "recyclable",
                "is_known": True,
                "usage": "known_train",
                "license_notes": "test",
            }
        ]
    )

    result = normalise_manifest_columns(df)

    assert "mapping_role" in result.columns
    assert "mapping_notes" in result.columns
    assert result["mapping_role"].iloc[0] == "known_train_candidate"


def test_validate_unknown_split_accepts_only_unknown_rows():
    unknown_df = pd.DataFrame(
        [
            make_manifest_row(
                sample_id="u1",
                source_dataset="realwaste",
                fine_label="unknown",
                coarse_label="unknown",
                is_known=False,
                usage="unknown_test",
                mapping_role="future_class_candidate",
            )
        ]
    )

    validate_unknown_split(
        unknown_df=unknown_df,
        unknown_fine_label="unknown",
        unknown_coarse_label="unknown",
    )


def test_build_combined_public_manifests_with_toy_data(tmp_path: Path):
    project_root = tmp_path

    split_dir = project_root / "ml" / "data" / "splits"
    config_path = project_root / "ml" / "configs" / "combined_public_training_manifests.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    label_rows = [
        ("paper_cardboard", "recyclable"),
        ("plastic", "recyclable"),
        ("glass", "recyclable"),
        ("metal", "recyclable"),
        ("organic", "organic"),
        ("residual", "residual"),
    ]

    for dataset_name in ["trashnet", "realwaste"]:
        for usage in ["known_train", "known_val", "known_test"]:
            rows = [
                make_manifest_row(
                    sample_id=f"{dataset_name}_{usage}_{index}",
                    source_dataset=dataset_name,
                    fine_label=fine_label,
                    coarse_label=coarse_label,
                    is_known=True,
                    usage=usage,
                )
                for index, (fine_label, coarse_label) in enumerate(label_rows)
            ]

            if dataset_name == "trashnet" and usage == "known_train":
                file_name = "known_train.csv"
            elif dataset_name == "trashnet" and usage == "known_val":
                file_name = "known_val.csv"
            elif dataset_name == "trashnet" and usage == "known_test":
                file_name = "known_test.csv"
            else:
                file_name = f"realwaste_{usage}_v1.csv"

            write_csv(split_dir / file_name, rows)

    unknown_rows = [
        make_manifest_row(
            sample_id="realwaste_unknown_1",
            source_dataset="realwaste",
            fine_label="unknown",
            coarse_label="unknown",
            is_known=False,
            usage="unknown_test",
            mapping_role="future_class_candidate",
        )
    ]
    write_csv(split_dir / "realwaste_unknown_test_v1.csv", unknown_rows)

    config = {
        "project": {
            "name": "OpenWaste-HR",
            "stage": "combined_public_training_manifests_test",
            "version": "v1",
        },
        "inputs": {
            "trashnet": {
                "known_train_csv": "ml/data/splits/known_train.csv",
                "known_val_csv": "ml/data/splits/known_val.csv",
                "known_test_csv": "ml/data/splits/known_test.csv",
            },
            "realwaste": {
                "known_train_csv": "ml/data/splits/realwaste_known_train_v1.csv",
                "known_val_csv": "ml/data/splits/realwaste_known_val_v1.csv",
                "known_test_csv": "ml/data/splits/realwaste_known_test_v1.csv",
                "unknown_test_csv": "ml/data/splits/realwaste_unknown_test_v1.csv",
            },
        },
        "outputs": {
            "output_dir": "ml/data/splits",
            "combined_manifest_csv": "expanded_public_manifest_v1.csv",
            "combined_known_train_csv": "expanded_public_known_train_v1.csv",
            "combined_known_val_csv": "expanded_public_known_val_v1.csv",
            "combined_known_test_csv": "expanded_public_known_test_v1.csv",
            "combined_unknown_test_csv": "expanded_public_unknown_test_v1.csv",
            "summary_json": "expanded_public_manifest_summary_v1.json",
        },
        "validation": {
            "require_known_train_labels": [
                "paper_cardboard",
                "plastic",
                "glass",
                "metal",
                "organic",
                "residual",
            ],
            "allowed_known_fine_labels": [
                "paper_cardboard",
                "plastic",
                "glass",
                "metal",
                "organic",
                "e_waste_battery",
                "residual",
            ],
            "unknown_fine_label": "unknown",
            "unknown_coarse_label": "unknown",
        },
    }

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = build_combined_public_manifests(
        config_path=config_path,
        project_root=project_root,
    )

    assert summary["known_train_samples"] == 12
    assert summary["known_val_samples"] == 12
    assert summary["known_test_samples"] == 12
    assert summary["unknown_samples"] == 1
    assert summary["total_combined_samples"] == 37

    combined_train = pd.read_csv(split_dir / "expanded_public_known_train_v1.csv")
    combined_unknown = pd.read_csv(split_dir / "expanded_public_unknown_test_v1.csv")

    assert set(combined_train["source_dataset"]) == {"trashnet", "realwaste"}
    assert set(combined_unknown["fine_label"]) == {"unknown"}
    assert set(combined_unknown["mapping_role"]) == {"future_class_candidate"}


def test_combined_public_outputs_exist_after_running():
    assert (
        PROJECT_ROOT / "ml" / "data" / "splits" / "expanded_public_manifest_v1.csv"
    ).exists()
    assert (
        PROJECT_ROOT / "ml" / "data" / "splits" / "expanded_public_known_train_v1.csv"
    ).exists()
    assert (
        PROJECT_ROOT / "ml" / "data" / "splits" / "expanded_public_unknown_test_v1.csv"
    ).exists()
    assert (
        PROJECT_ROOT / "ml" / "data" / "splits" / "expanded_public_manifest_summary_v1.json"
    ).exists()
