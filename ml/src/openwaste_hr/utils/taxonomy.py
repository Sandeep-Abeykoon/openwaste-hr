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


def _uses_current_taxonomy_schema(taxonomy: dict[str, Any]) -> bool:
    return "known_classes" in taxonomy and "coarse_classes" in taxonomy


def _normalise_string_list(values: Any, field_name: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"Taxonomy field '{field_name}' must be a list.")

    normalised_values = [str(value).strip() for value in values]
    if any(not value for value in normalised_values):
        raise ValueError(f"Taxonomy field '{field_name}' contains an empty label.")

    return normalised_values


def _get_current_coarse_members(taxonomy: dict[str, Any]) -> dict[str, list[str]]:
    coarse_classes = taxonomy.get("coarse_classes", {})

    if not isinstance(coarse_classes, dict):
        raise ValueError("Taxonomy field 'coarse_classes' must be a dictionary.")

    normalised_mapping: dict[str, list[str]] = {}
    for coarse_label, members in coarse_classes.items():
        normalised_coarse_label = str(coarse_label).strip()
        if not normalised_coarse_label:
            raise ValueError("Taxonomy field 'coarse_classes' contains an empty key.")

        normalised_mapping[normalised_coarse_label] = _normalise_string_list(
            members,
            f"coarse_classes.{normalised_coarse_label}",
        )

    return normalised_mapping


def _get_unknown_evaluation_labels(taxonomy: dict[str, Any]) -> list[str]:
    unknown_classes = taxonomy.get("unknown_evaluation_classes", {})

    if unknown_classes in ({}, None):
        return []

    if not isinstance(unknown_classes, dict):
        raise ValueError(
            "Taxonomy field 'unknown_evaluation_classes' must be a dictionary."
        )

    canonical_labels: list[str] = []
    for entry_name, entry in unknown_classes.items():
        if not isinstance(entry, dict):
            raise ValueError(
                "Each unknown_evaluation_classes entry must be a dictionary."
            )

        canonical_name = str(entry.get("canonical_name", entry_name)).strip()
        if not canonical_name:
            raise ValueError(
                "Unknown evaluation taxonomy entries must define a canonical_name."
            )

        aliases = entry.get("aliases", [])
        _normalise_string_list(aliases, f"unknown_evaluation_classes.{entry_name}.aliases")
        canonical_labels.append(canonical_name)

    return canonical_labels


def get_fine_labels(taxonomy: dict[str, Any]) -> list[str]:
    """
    Return known fine label names in taxonomy order.
    """
    if _uses_current_taxonomy_schema(taxonomy):
        return _normalise_string_list(taxonomy.get("known_classes", []), "known_classes")

    fine_labels = taxonomy.get("fine_labels", [])
    sorted_labels = sorted(fine_labels, key=lambda item: item["id"])
    return [item["name"] for item in sorted_labels]


def get_coarse_labels(taxonomy: dict[str, Any]) -> list[str]:
    """
    Return coarse label names in taxonomy order.
    """
    if _uses_current_taxonomy_schema(taxonomy):
        return list(_get_current_coarse_members(taxonomy).keys())

    coarse_labels = taxonomy.get("coarse_labels", [])
    sorted_labels = sorted(coarse_labels, key=lambda item: item["id"])
    return [item["name"] for item in sorted_labels]


def get_fine_to_coarse_map(taxonomy: dict[str, Any]) -> dict[str, str]:
    """
    Return a mapping from known fine label name to coarse label name.
    """
    if _uses_current_taxonomy_schema(taxonomy):
        known_classes = get_fine_labels(taxonomy)
        coarse_members = _get_current_coarse_members(taxonomy)
        mapping: dict[str, str] = {}

        for label in known_classes:
            matching_coarse_labels = [
                coarse_label
                for coarse_label, members in coarse_members.items()
                if label in members
            ]

            if len(matching_coarse_labels) != 1:
                raise ValueError(
                    f"Known class '{label}' must appear in exactly one coarse class."
                )

            mapping[label] = matching_coarse_labels[0]

        return mapping

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
    if _uses_current_taxonomy_schema(taxonomy):
        known_classes = get_fine_labels(taxonomy)
        coarse_labels = get_coarse_labels(taxonomy)
        fine_to_coarse = get_fine_to_coarse_map(taxonomy)
        unknown_evaluation_labels = _get_unknown_evaluation_labels(taxonomy)
        excluded_classes = set(
            _normalise_string_list(taxonomy.get("excluded_classes", []), "excluded_classes")
        )
        coarse_members = _get_current_coarse_members(taxonomy)

        if len(known_classes) != len(set(known_classes)):
            raise ValueError("Duplicate known_classes entries found.")

        if len(coarse_labels) != len(set(coarse_labels)):
            raise ValueError("Duplicate coarse class names found.")

        if "unknown" in set(known_classes):
            raise ValueError("Unknown must not be included as a known fine training label.")

        if set(known_classes).intersection(excluded_classes):
            raise ValueError("Known classes must not overlap with excluded classes.")

        if len(unknown_evaluation_labels) != len(set(unknown_evaluation_labels)):
            raise ValueError("Duplicate unknown evaluation canonical labels found.")

        if set(unknown_evaluation_labels).intersection(known_classes):
            raise ValueError(
                "Unknown evaluation classes must not overlap with known_classes."
            )

        flattened_members = [
            member
            for members in coarse_members.values()
            for member in members
        ]
        if len(flattened_members) != len(set(flattened_members)):
            raise ValueError("A class appears in more than one coarse class.")

        missing_known_classes = [
            label for label in known_classes
            if label not in fine_to_coarse
        ]
        if missing_known_classes:
            raise ValueError(
                f"Known classes are missing from coarse_classes: {missing_known_classes}"
            )

        missing_unknown_labels = [
            label for label in unknown_evaluation_labels
            if label not in flattened_members
        ]
        if missing_unknown_labels:
            raise ValueError(
                "Unknown evaluation classes are missing from coarse_classes: "
                f"{missing_unknown_labels}"
            )

        return True

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
