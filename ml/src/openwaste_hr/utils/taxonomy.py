from pathlib import Path
from typing import Any

import yaml


def load_taxonomy(config_path: str | Path) -> dict[str, Any]:
    """
    Load the OpenWaste-HR taxonomy YAML file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        taxonomy = yaml.safe_load(file)

    if not isinstance(taxonomy, dict):
        raise ValueError("Taxonomy file must contain a YAML dictionary.")

    return taxonomy


def get_fine_labels(taxonomy: dict[str, Any]) -> list[str]:
    """
    Return known fine label names in ID order.
    """
    fine_labels = taxonomy.get("fine_labels", [])
    sorted_labels = sorted(fine_labels, key=lambda item: item["id"])
    return [item["name"] for item in sorted_labels]


def get_coarse_labels(taxonomy: dict[str, Any]) -> list[str]:
    """
    Return coarse label names in ID order.
    """
    coarse_labels = taxonomy.get("coarse_labels", [])
    sorted_labels = sorted(coarse_labels, key=lambda item: item["id"])
    return [item["name"] for item in sorted_labels]


def get_fine_to_coarse_map(taxonomy: dict[str, Any]) -> dict[str, str]:
    """
    Return a mapping from fine label name to coarse label name.
    """
    fine_labels = taxonomy.get("fine_labels", [])
    return {item["name"]: item["coarse_name"] for item in fine_labels}


def validate_taxonomy(taxonomy: dict[str, Any]) -> bool:
    """
    Validate important taxonomy rules.

    Rules:
    1. Fine IDs must be unique.
    2. Coarse IDs must be unique.
    3. Every fine label must point to an existing coarse ID.
    4. Unknown must not be included as a known fine label.
    """
    fine_labels = taxonomy.get("fine_labels", [])
    coarse_labels = taxonomy.get("coarse_labels", [])

    fine_ids = [item["id"] for item in fine_labels]
    coarse_ids = [item["id"] for item in coarse_labels]
    coarse_id_set = set(coarse_ids)

    if len(fine_ids) != len(set(fine_ids)):
        raise ValueError("Duplicate fine label IDs found.")

    if len(coarse_ids) != len(set(coarse_ids)):
        raise ValueError("Duplicate coarse label IDs found.")

    for item in fine_labels:
        if item["coarse_id"] not in coarse_id_set:
            raise ValueError(
                f"Fine label '{item['name']}' points to missing coarse_id {item['coarse_id']}."
            )

    fine_names = {item["name"] for item in fine_labels}
    if "unknown" in fine_names:
        raise ValueError("Unknown must not be included as a known fine training label.")

    return True