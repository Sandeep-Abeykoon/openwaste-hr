from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_public_dataset_expansion_plan_files_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "public_dataset_expansion_plan_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "public_dataset_expansion_plan_summary_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT / "ml" / "configs" / "public_dataset_expansion_plan.yaml"
    ).exists()


def test_public_dataset_expansion_plan_mentions_current_limitation():
    text = read_text("docs/methodology/public_dataset_expansion_plan_v1.md")

    assert "TrashNet-style dataset" in text
    assert "small and limited" in text
    assert "rubber slipper" in text
    assert "local_000001" in text


def test_public_dataset_expansion_plan_mentions_candidate_datasets():
    text = read_text("docs/methodology/public_dataset_expansion_plan_v1.md")

    assert "RealWaste" in text
    assert "TACO" in text
    assert "expanded public dataset" in text
    assert "pretrained expanded model" in text


def test_public_dataset_expansion_plan_mentions_mapping_safety():
    text = read_text("docs/methodology/public_dataset_expansion_plan_v1.md")

    assert "Do not force unclear or outside-taxonomy labels into known classes" in text
    assert "unknown_eval_candidate" in text
    assert "future_class_candidate" in text
    assert "exclude_or_review" in text


def test_public_dataset_expansion_yaml_contains_realwaste_and_taco():
    config_path = PROJECT_ROOT / "ml" / "configs" / "public_dataset_expansion_plan.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_ids = {
        dataset["dataset_id"] for dataset in data["candidate_datasets"]
    }

    assert "realwaste" in dataset_ids
    assert "taco" in dataset_ids


def test_public_dataset_expansion_yaml_contains_mapping_rules():
    config_path = PROJECT_ROOT / "ml" / "configs" / "public_dataset_expansion_plan.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert data["mapping_rules"]["do_not_force_unknown_into_known_class"] is True
    assert data["mapping_rules"]["require_manual_mapping_review"] is True
    assert "known_train_candidate" in data["mapping_roles"]
    assert "unknown_eval_candidate" in data["mapping_roles"]


def test_supervisor_summary_mentions_realwaste_next():
    text = read_text("docs/supervisor_updates/public_dataset_expansion_plan_summary_v1.md")

    assert "RealWaste" in text
    assert "next implementation step" in text.lower()
    assert "manifest builder" in text
    assert "current best pretrained safe policy" in text