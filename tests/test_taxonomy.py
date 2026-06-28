import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(SRC_PATH))

from openwaste_hr.utils.taxonomy import (  # noqa: E402
    get_coarse_labels,
    get_fine_labels,
    get_fine_to_coarse_map,
    load_taxonomy,
    validate_taxonomy,
)


def test_taxonomy_loads_successfully():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"
    taxonomy = load_taxonomy(taxonomy_path)

    assert taxonomy["project"]["name"] == "OpenWaste-HR"
    assert validate_taxonomy(taxonomy) is True


def test_fine_labels_are_correct():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"
    taxonomy = load_taxonomy(taxonomy_path)

    fine_labels = get_fine_labels(taxonomy)

    assert fine_labels == [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
    ]


def test_coarse_labels_are_correct():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"
    taxonomy = load_taxonomy(taxonomy_path)

    coarse_labels = get_coarse_labels(taxonomy)

    assert coarse_labels == [
        "recyclable",
        "unknown",
    ]


def test_fine_to_coarse_mapping_is_correct():
    taxonomy_path = PROJECT_ROOT / "ml" / "configs" / "taxonomy.yaml"
    taxonomy = load_taxonomy(taxonomy_path)

    mapping = get_fine_to_coarse_map(taxonomy)

    assert mapping["cardboard"] == "recyclable"
    assert mapping["plastic"] == "recyclable"
    assert mapping["glass"] == "recyclable"
    assert mapping["metal"] == "recyclable"
    assert mapping["paper"] == "recyclable"
