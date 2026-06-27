from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def normalize_uploaded_filename(value: Any) -> str:
    return normalize_text(value).casefold()


def get_manual_review_dir(repo_root: Path) -> Path:
    return repo_root / "ml" / "outputs" / "manual_review"


def get_review_queue_path(repo_root: Path) -> Path:
    return get_manual_review_dir(repo_root) / "review_queue.json"


def get_intelligence_candidates_path(repo_root: Path) -> Path:
    return get_manual_review_dir(repo_root) / "intelligence_candidates.json"


def get_api_uploads_dir(repo_root: Path) -> Path:
    return repo_root / "ml" / "outputs" / "api_uploads"


def load_review_queue(queue_path: Path) -> list[dict[str, Any]]:
    if not queue_path.exists():
        return []

    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Manual review queue must be a JSON array.")

    normalized_items: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            normalized_items.append(item)

    return normalized_items


def save_review_queue(queue_path: Path, items: list[dict[str, Any]]) -> None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(items, indent=2),
        encoding="utf-8",
    )


def build_intelligence_candidates(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for item in items:
        review = item.get("review") or {}
        if not review.get("promote_to_intelligence"):
            continue

        result = item.get("result") or {}
        prediction = result.get("prediction") or {}
        fusion_gate = result.get("fusion_gate") or {}

        candidates.append(
            {
                "review_id": item.get("review_id"),
                "uploaded_filename": item.get("uploaded_filename"),
                "stored_image_path": item.get("stored_image_path"),
                "review_label": review.get("review_label"),
                "review_notes": review.get("review_notes"),
                "reviewed_at": review.get("reviewed_at"),
                "internal_top1_prediction": prediction.get(
                    "internal_top1_prediction"
                ),
                "knownness_score": fusion_gate.get("knownness_score"),
                "threshold": fusion_gate.get("threshold"),
            }
        )

    return candidates


def save_intelligence_candidates(
    repo_root: Path,
    items: list[dict[str, Any]],
) -> None:
    candidates_path = get_intelligence_candidates_path(repo_root)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.write_text(
        json.dumps(build_intelligence_candidates(items), indent=2),
        encoding="utf-8",
    )


def is_path_inside_directory(candidate_path: Path, directory_path: Path) -> bool:
    resolved_candidate = candidate_path.resolve()
    resolved_directory = directory_path.resolve()
    return (
        resolved_candidate == resolved_directory
        or resolved_directory in resolved_candidate.parents
    )


def sort_review_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        status_rank = 1 if item.get("status") == "pending" else 0
        timestamp = item.get("updated_at") or item.get("created_at") or ""
        return (status_rank, timestamp)

    return sorted(items, key=sort_key, reverse=True)


def summarize_review_queue(items: list[dict[str, Any]]) -> dict[str, int]:
    pending_count = 0
    reviewed_count = 0
    intelligence_count = 0

    for item in items:
        if item.get("status") == "reviewed":
            reviewed_count += 1
        else:
            pending_count += 1

        review = item.get("review") or {}
        if review.get("promote_to_intelligence"):
            intelligence_count += 1

    return {
        "pending_count": pending_count,
        "reviewed_count": reviewed_count,
        "intelligence_count": intelligence_count,
        "total_count": len(items),
    }


def create_review_entry(
    *,
    review_id: str,
    response_payload: dict[str, Any],
) -> dict[str, Any]:
    result = response_payload.get("result") or {}
    prediction = result.get("prediction") or {}
    fusion_gate = result.get("fusion_gate") or {}
    final_decision = result.get("final_decision") or {}
    created_at = utc_timestamp()

    return {
        "review_id": review_id,
        "status": "pending",
        "created_at": created_at,
        "updated_at": created_at,
        "uploaded_filename": response_payload.get("uploaded_filename"),
        "stored_image_path": response_payload.get("stored_image_path"),
        "result": result,
        "review": None,
        "summary": {
            "internal_top1_prediction": prediction.get("internal_top1_prediction"),
            "knownness_score": fusion_gate.get("knownness_score"),
            "threshold": fusion_gate.get("threshold"),
            "decision_type": final_decision.get("decision_type"),
            "user_message": final_decision.get("user_message"),
        },
    }


def queue_manual_review(
    *,
    repo_root: Path,
    review_id: str,
    response_payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    queue_path = get_review_queue_path(repo_root)
    items = load_review_queue(queue_path)
    existing_item = find_review_item_by_uploaded_filename(
        items,
        response_payload.get("uploaded_filename"),
    )

    if existing_item is not None:
        delete_uploaded_image_if_unused(
            repo_root=repo_root,
            stored_image_path=response_payload.get("stored_image_path"),
            remaining_items=items,
        )
        return existing_item, False

    entry = create_review_entry(
        review_id=review_id,
        response_payload=response_payload,
    )
    items.append(entry)
    save_review_queue(queue_path, items)
    save_intelligence_candidates(repo_root, items)
    return entry, True


def list_manual_reviews(repo_root: Path) -> dict[str, Any]:
    queue_path = get_review_queue_path(repo_root)
    items = sort_review_items(load_review_queue(queue_path))
    return {
        "items": items,
        "summary": summarize_review_queue(items),
    }


def find_review_item(
    items: list[dict[str, Any]],
    review_id: str,
) -> tuple[int, dict[str, Any]]:
    for index, item in enumerate(items):
        if item.get("review_id") == review_id:
            return index, item

    raise KeyError(f"Manual review item not found: {review_id}")


def find_review_item_by_uploaded_filename(
    items: list[dict[str, Any]],
    uploaded_filename: str | None,
) -> dict[str, Any] | None:
    normalized_uploaded_filename = normalize_uploaded_filename(uploaded_filename)

    if normalized_uploaded_filename == "":
        return None

    for item in items:
        if (
            normalize_uploaded_filename(item.get("uploaded_filename"))
            == normalized_uploaded_filename
        ):
            return item

    return None


def delete_uploaded_image_if_unused(
    *,
    repo_root: Path,
    stored_image_path: str | None,
    remaining_items: list[dict[str, Any]],
) -> bool:
    normalized_stored_image_path = normalize_text(stored_image_path)
    if normalized_stored_image_path == "":
        return False

    image_path = Path(normalized_stored_image_path).resolve()
    uploads_dir = get_api_uploads_dir(repo_root)

    if not is_path_inside_directory(image_path, uploads_dir):
        return False

    for item in remaining_items:
        item_image_path = normalize_text(item.get("stored_image_path"))
        if item_image_path != "" and Path(item_image_path).resolve() == image_path:
            return False

    if image_path.exists():
        image_path.unlink()
        return True

    return False


def resolve_review_image_path(repo_root: Path, review_id: str) -> Path:
    queue_path = get_review_queue_path(repo_root)
    items = load_review_queue(queue_path)
    _, item = find_review_item(items, review_id)
    image_path = Path(item["stored_image_path"]).resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Stored manual review image not found: {image_path}"
        )

    return image_path


def update_review_decision(
    *,
    repo_root: Path,
    review_id: str,
    selected_label: str | None = None,
    custom_label: str | None = None,
    review_notes: str | None = None,
    promote_to_intelligence: bool = False,
) -> dict[str, Any]:
    queue_path = get_review_queue_path(repo_root)
    items = load_review_queue(queue_path)
    item_index, item = find_review_item(items, review_id)

    normalized_selected_label = normalize_text(selected_label)
    normalized_custom_label = normalize_text(custom_label)
    normalized_review_notes = normalize_text(review_notes)
    review_label = normalized_custom_label or normalized_selected_label

    if review_label == "":
        raise ValueError("A review label is required.")

    reviewed_at = utc_timestamp()

    updated_item = {
        **item,
        "status": "reviewed",
        "updated_at": reviewed_at,
        "review": {
            "selected_label": normalized_selected_label or None,
            "custom_label": normalized_custom_label or None,
            "review_label": review_label,
            "review_notes": normalized_review_notes or None,
            "promote_to_intelligence": bool(promote_to_intelligence),
            "reviewed_at": reviewed_at,
        },
    }

    items[item_index] = updated_item
    save_review_queue(queue_path, items)
    save_intelligence_candidates(repo_root, items)
    return updated_item


def delete_manual_review(
    *,
    repo_root: Path,
    review_id: str,
) -> tuple[dict[str, Any], bool]:
    queue_path = get_review_queue_path(repo_root)
    items = load_review_queue(queue_path)
    item_index, item = find_review_item(items, review_id)
    remaining_items = [
        existing_item
        for existing_index, existing_item in enumerate(items)
        if existing_index != item_index
    ]

    deleted_image_from_system = delete_uploaded_image_if_unused(
        repo_root=repo_root,
        stored_image_path=item.get("stored_image_path"),
        remaining_items=remaining_items,
    )

    save_review_queue(queue_path, remaining_items)
    save_intelligence_candidates(repo_root, remaining_items)
    return item, deleted_image_from_system
