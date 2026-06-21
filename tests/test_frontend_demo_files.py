from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_frontend_files_exist():
    assert (PROJECT_ROOT / "frontend" / "index.html").exists()
    assert (PROJECT_ROOT / "frontend" / "styles.css").exists()
    assert (PROJECT_ROOT / "frontend" / "app.js").exists()


def test_index_contains_required_inputs_and_script():
    html = read_text("frontend/index.html")

    assert 'id="predictionForm"' in html
    assert 'id="imagePathInput"' in html
    assert 'id="sampleIdInput"' in html
    assert 'id="apiBaseInput"' in html
    assert 'id="predictBtn"' in html
    assert 'src="app.js"' in html


def test_app_js_calls_backend_inference_endpoint():
    app_js = read_text("frontend/app.js")

    assert "/api/inference/predict" in app_js
    assert "fetch(" in app_js
    assert "renderProbabilityTable" in app_js
    assert "frontend_demo_" in app_js


def test_styles_include_demo_cards():
    css = read_text("frontend/styles.css")

    assert ".decision-card" in css
    assert ".raw-json" in css
    assert ".status-message" in css


def test_backend_main_contains_cors_middleware():
    main_py = read_text("backend/app/main.py")

    assert "CORSMiddleware" in main_py
    assert "allow_origins" in main_py