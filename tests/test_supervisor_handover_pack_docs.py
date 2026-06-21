from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_supervisor_handover_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "supervisor_handover_pack_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "supervisor_demo_script_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "supervisor_final_status_summary_v1.md"
    ).exists()


def test_handover_pack_mentions_best_project_message():
    text = read_text("docs/supervisor_updates/supervisor_handover_pack_v1.md")

    assert "pretrained image recognition" in text
    assert "hierarchical open-set decision-making" in text
    assert "local unknown evaluation" in text
    assert "human-in-the-loop active learning" in text


def test_handover_pack_mentions_best_policy_metrics():
    text = read_text("docs/supervisor_updates/supervisor_handover_pack_v1.md")

    assert "Pretrained Safe Hierarchical Policy" in text
    assert "0.864583" in text
    assert "0.960843" in text
    assert "0.600000" in text
    assert "0.400000" in text


def test_handover_pack_mentions_local_unknown_example():
    text = read_text("docs/supervisor_updates/supervisor_handover_pack_v1.md")

    assert "local_000001" in text
    assert "rubber slipper" in text
    assert "flip-flop" in text
    assert "outside_current_known_taxonomy" in text
    assert "unknown_test_and_future_class_candidate" in text


def test_demo_script_mentions_backend_and_frontend_commands():
    text = read_text("docs/supervisor_updates/supervisor_demo_script_v1.md")

    assert "uvicorn backend.app.main:app" in text
    assert "python -m http.server 5500" in text
    assert "http://127.0.0.1:5500" in text
    assert "ml/data/local_unknown/images/local_000001.jpg" in text


def test_final_status_mentions_pipeline_components():
    text = read_text("docs/supervisor_updates/supervisor_final_status_summary_v1.md")

    assert "inference pipeline" in text
    assert "FastAPI backend" in text
    assert "frontend demo" in text
    assert "best policy integrated" in text


def test_final_status_mentions_test_status():
    text = read_text("docs/supervisor_updates/supervisor_final_status_summary_v1.md")

    assert "241 passed" in text
    assert "1 warning" in text