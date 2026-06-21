from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_human_correction_label_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "human_correction_label_preparation_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "human_correction_label_preparation_summary_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "annotation"
        / "local_unknown_human_label_guide_v1.md"
    ).exists()


def test_human_correction_methodology_mentions_local_unknown_example():
    text = read_text("docs/methodology/human_correction_label_preparation_v1.md")

    assert "local_000001" in text
    assert "rubber slipper" in text
    assert "flip-flop" in text
    assert "outside_current_known_taxonomy" in text
    assert "keep_as_unknown_test" in text


def test_human_correction_methodology_mentions_active_learning_v2():
    text = read_text("docs/methodology/human_correction_label_preparation_v1.md")

    assert "active learning v2" in text.lower()
    assert "human correction labels" in text
    assert "manual-review" in text
    assert "hierarchical open-set waste classification" in text


def test_human_label_guide_mentions_known_labels_and_actions():
    text = read_text("docs/annotation/local_unknown_human_label_guide_v1.md")

    assert "paper_cardboard" in text
    assert "plastic" in text
    assert "glass" in text
    assert "metal" in text
    assert "organic" in text
    assert "e_waste_battery" in text
    assert "keep_as_unknown_test" in text
    assert "create_future_class_candidate" in text


def test_human_observation_notes_csv_exists_and_records_example():
    path = PROJECT_ROOT / "ml" / "data" / "splits" / "local_unknown_human_observation_notes_v1.csv"
    assert path.exists()

    text = path.read_text(encoding="utf-8")

    assert "local_000001" in text
    assert "rubber slipper" in text
    assert "outside_current_known_taxonomy" in text
    assert "keep_as_unknown_test" in text


def test_supervisor_summary_mentions_human_in_loop_value():
    text = read_text(
        "docs/supervisor_updates/human_correction_label_preparation_summary_v1.md"
    )

    assert "human correction labels" in text
    assert "active learning v2" in text.lower()
    assert "local unknown evaluation" in text
    assert "human-in-the-loop active learning" in text