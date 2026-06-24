from pathlib import Path


def test_readme_exists_and_uses_final_fusion_gate_story():
    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md is missing."

    text = readme_path.read_text(encoding="utf-8")

    required_phrases = [
        "Fusion Gate v2",
        "Mahalanobis",
        "false acceptance",
        "0.0663",
        "0.9337",
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
    ]

    for phrase in required_phrases:
        assert phrase in text, f"README.md is missing required phrase: {phrase}"


def test_final_result_documents_exist():
    required_files = [
        Path("docs/results/openwaste_hr_final_results_summary_v1.md"),
        Path("docs/results/fusion_gate_v2_mahalanobis_results_v1.md"),
        Path("docs/results/fusion_gate_v2_statistical_evaluation_v1.md"),
        Path("docs/methodology/final_decision_policy_v2_fusion_gate.md"),
        Path("ml/configs/final_decision_policy_v2_fusion_gate.yaml"),
    ]

    for path in required_files:
        assert path.exists(), f"Required final project file is missing: {path}"


def test_final_policy_config_mentions_fusion_gate_v2():
    policy_path = Path("ml/configs/final_decision_policy_v2_fusion_gate.yaml")
    assert policy_path.exists(), "Final Fusion Gate v2 policy config is missing."

    text = policy_path.read_text(encoding="utf-8")

    assert "fusion_gate_v2_mahalanobis_v1" in text
    assert "0.6314586412215439" in text
    assert "mahalanobis" in text.lower()
    assert "manual_review" in text
