from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_cleanup_summary_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "readme_consistency_cleanup_summary_v1.md"
    ).exists()


def test_readme_has_final_best_balanced_system():
    text = read_text("README.md")

    assert "Baseline C + Safe Policy C" in text
    assert "pretrained expanded public model" in text
    assert "expanded public safe hierarchical decision policy" in text


def test_readme_no_longer_claims_old_current_best_system():
    text = read_text("README.md")

    assert "## Current Best System" not in text
    assert "The current best system is:" not in text
    assert "Pretrained Safe Hierarchical Policy" not in text


def test_readme_latest_test_count_is_updated():
    text = read_text("README.md")

    assert "330 passed, 1 warning" in text
    assert "248 passed, 1 warning" not in text


def test_readme_preserves_local_unknown_and_active_learning_notes():
    text = read_text("README.md")

    assert "Local Unknown Example" in text
    assert "rubber slipper / flip-flop" in text
    assert "Active Learning v2" in text
    assert "unknown_test_and_future_class_candidate" in text


def test_readme_cleanup_summary_mentions_reason():
    text = read_text("docs/supervisor_updates/readme_consistency_cleanup_summary_v1.md")

    assert "Current Best System" in text
    assert "TrashNet-only safe hierarchical policy" in text
    assert "Baseline C + Safe Policy C" in text