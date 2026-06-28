from pathlib import Path
from tempfile import TemporaryDirectory

from ml.api.manual_review_queue import (
    delete_manual_review,
    get_intelligence_candidates_path,
    get_review_queue_path,
    list_manual_reviews,
    queue_manual_review,
    resolve_review_image_path,
    update_review_decision,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_response_payload(
    image_path: Path,
    uploaded_filename: str = "review-sample.jpg",
) -> dict:
    return {
        "uploaded_filename": uploaded_filename,
        "stored_image_path": str(image_path),
        "result": {
            "prediction": {
                "internal_top1_prediction": "paper",
            },
            "fusion_gate": {
                "knownness_score": 0.2,
                "threshold": 0.63,
            },
            "final_decision": {
                "accepted_as_known": False,
                "decision_type": "unknown_manual_review",
                "user_message": "Please review this sample.",
            },
        },
    }


def test_queue_manual_review_and_submit_decision():
    temp_root = PROJECT_ROOT / "tmp_test_runs"
    temp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=temp_root) as temp_dir:
        repo_root = Path(temp_dir)
        image_path = repo_root / "ml" / "outputs" / "api_uploads" / "review.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")

        review_entry, was_created = queue_manual_review(
            repo_root=repo_root,
            review_id="review-001",
            response_payload=create_response_payload(image_path),
        )

        assert was_created is True
        assert review_entry["status"] == "pending"
        assert review_entry["summary"]["internal_top1_prediction"] == "paper"

        queue_payload = list_manual_reviews(repo_root)
        assert queue_payload["summary"]["pending_count"] == 1
        assert queue_payload["summary"]["reviewed_count"] == 0
        assert queue_payload["items"][0]["review_id"] == "review-001"

        updated_item = update_review_decision(
            repo_root=repo_root,
            review_id="review-001",
            selected_label="paper",
            custom_label="",
            review_notes="Looks like mixed paper waste.",
            promote_to_intelligence=True,
        )

        assert updated_item["status"] == "reviewed"
        assert updated_item["review"]["review_label"] == "paper"
        assert updated_item["review"]["promote_to_intelligence"] is True

        updated_queue_payload = list_manual_reviews(repo_root)
        assert updated_queue_payload["summary"]["pending_count"] == 0
        assert updated_queue_payload["summary"]["reviewed_count"] == 1
        assert updated_queue_payload["summary"]["intelligence_count"] == 1

        intelligence_candidates = get_intelligence_candidates_path(repo_root)
        assert intelligence_candidates.exists()
        assert '"review_label": "paper"' in intelligence_candidates.read_text(
            encoding="utf-8"
        )

        resolved_image_path = resolve_review_image_path(repo_root, "review-001")
        assert resolved_image_path == image_path.resolve()


def test_update_review_requires_label():
    temp_root = PROJECT_ROOT / "tmp_test_runs"
    temp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=temp_root) as temp_dir:
        repo_root = Path(temp_dir)
        image_path = repo_root / "ml" / "outputs" / "api_uploads" / "review.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")

        _, was_created = queue_manual_review(
            repo_root=repo_root,
            review_id="review-002",
            response_payload=create_response_payload(image_path),
        )
        assert was_created is True

        try:
            update_review_decision(
                repo_root=repo_root,
                review_id="review-002",
                selected_label="",
                custom_label="",
                review_notes="",
                promote_to_intelligence=False,
            )
        except ValueError as error:
            assert str(error) == "A review label is required."
        else:
            raise AssertionError("Expected ValueError when no review label is given.")

        assert get_review_queue_path(repo_root).exists()


def test_queue_manual_review_skips_duplicate_uploaded_filename():
    temp_root = PROJECT_ROOT / "tmp_test_runs"
    temp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=temp_root) as temp_dir:
        repo_root = Path(temp_dir)
        uploads_dir = repo_root / "ml" / "outputs" / "api_uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        first_image_path = uploads_dir / "review-1.jpg"
        first_image_path.write_bytes(b"first-image")
        second_image_path = uploads_dir / "review-2.jpg"
        second_image_path.write_bytes(b"second-image")

        first_entry, first_created = queue_manual_review(
            repo_root=repo_root,
            review_id="review-003",
            response_payload=create_response_payload(
                first_image_path,
                uploaded_filename="duplicate-name.jpg",
            ),
        )
        second_entry, second_created = queue_manual_review(
            repo_root=repo_root,
            review_id="review-004",
            response_payload=create_response_payload(
                second_image_path,
                uploaded_filename="duplicate-name.jpg",
            ),
        )

        assert first_created is True
        assert second_created is False
        assert first_entry["review_id"] == second_entry["review_id"] == "review-003"
        assert not second_image_path.exists()

        queue_payload = list_manual_reviews(repo_root)
        assert queue_payload["summary"]["total_count"] == 1


def test_delete_manual_review_removes_item_and_stored_image():
    temp_root = PROJECT_ROOT / "tmp_test_runs"
    temp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=temp_root) as temp_dir:
        repo_root = Path(temp_dir)
        image_path = repo_root / "ml" / "outputs" / "api_uploads" / "review-delete.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")

        _, was_created = queue_manual_review(
            repo_root=repo_root,
            review_id="review-005",
            response_payload=create_response_payload(
                image_path,
                uploaded_filename="delete-me.jpg",
            ),
        )
        assert was_created is True

        deleted_item, deleted_image_from_system = delete_manual_review(
            repo_root=repo_root,
            review_id="review-005",
        )

        assert deleted_item["review_id"] == "review-005"
        assert deleted_image_from_system is True
        assert not image_path.exists()

        queue_payload = list_manual_reviews(repo_root)
        assert queue_payload["summary"]["total_count"] == 0
