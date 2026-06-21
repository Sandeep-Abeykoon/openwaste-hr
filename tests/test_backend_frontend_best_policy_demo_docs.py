from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_backend_frontend_demo_docs_exist():
    assert (
        PROJECT_ROOT
        / "docs"
        / "methodology"
        / "backend_frontend_best_policy_demo_test_v1.md"
    ).exists()

    assert (
        PROJECT_ROOT
        / "docs"
        / "supervisor_updates"
        / "backend_frontend_best_policy_demo_test_summary_v1.md"
    ).exists()


def test_backend_frontend_demo_methodology_mentions_best_policy():
    text = read_text("docs/methodology/backend_frontend_best_policy_demo_test_v1.md")

    assert "Pretrained Safe Hierarchical Policy" in text
    assert "pretrained_trashnet_v1_best.pt" in text
    assert "0.995000" in text
    assert "0.900000" in text


def test_backend_frontend_demo_methodology_mentions_components():
    text = read_text("docs/methodology/backend_frontend_best_policy_demo_test_v1.md")

    assert "backend health endpoint" in text
    assert "backend prediction endpoint" in text
    assert "frontend demo page" in text
    assert "local unknown image" in text


def test_backend_frontend_demo_methodology_mentions_rubber_slipper():
    text = read_text("docs/methodology/backend_frontend_best_policy_demo_test_v1.md")

    assert "rubber slipper" in text
    assert "flip-flop" in text
    assert "local unknown" in text


def test_backend_frontend_demo_summary_mentions_thesis_value():
    text = read_text(
        "docs/supervisor_updates/backend_frontend_best_policy_demo_test_summary_v1.md"
    )

    assert "live backend and frontend demo" in text
    assert "selected research policy" in text
    assert "closed-set confidence" in text
    assert "open-world waste classification" in text


def test_frontend_contains_best_policy_text():
    text = read_text("frontend/index.html")

    assert "OpenWaste-HR Best Prototype" in text
    assert "pretrained safe hierarchical policy" in text