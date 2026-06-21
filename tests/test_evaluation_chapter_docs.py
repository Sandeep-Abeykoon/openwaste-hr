from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_evaluation_chapter_draft_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "thesis"
        / "evaluation_chapter_draft_v1.md"
    ).exists()


def test_evaluation_summary_exists():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "evaluation_chapter_summary_v1.md"
    ).exists()


def test_evaluation_chapter_mentions_core_results():
    text = read_text("docs/thesis/evaluation_chapter_draft_v1.md")

    assert "0.692708" in text
    assert "0.770992" in text
    assert "0.375000" in text
    assert "0.962933" in text


def test_evaluation_chapter_mentions_policy_comparison():
    text = read_text("docs/thesis/evaluation_chapter_draft_v1.md")

    assert "Closed-set baseline" in text
    assert "Confidence reject baseline" in text
    assert "Hierarchical policy v1" in text
    assert "Safe hierarchical policy" in text


def test_evaluation_chapter_mentions_future_evaluation_plan():
    text = read_text("docs/thesis/evaluation_chapter_draft_v1.md")

    assert "Pretrained model training" in text
    assert "Additional public datasets" in text
    assert "Human-labelled local samples" in text
    assert "Active-learning v2 model" in text


def test_evaluation_summary_mentions_not_final_stage():
    text = read_text("docs/supervisor_updates/evaluation_chapter_summary_v1.md")

    assert "not the final comparison stage" in text
    assert "pretrained transfer-learning model" in text
    assert "reviewed local dataset v2" in text