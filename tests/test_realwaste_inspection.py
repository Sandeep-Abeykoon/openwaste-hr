from pathlib import Path

import pandas as pd

from openwaste_hr.data.inspect_realwaste_manifest import (
    grouped_counts_table,
    value_counts_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_realwaste_inspection_script_exists():
    assert (
        PROJECT_ROOT / "ml" / "src" / "openwaste_hr" / "data" / "inspect_realwaste_manifest.py"
    ).exists()


def test_value_counts_table_returns_counts_and_proportions():
    df = pd.DataFrame({"label": ["a", "a", "b"]})

    result = value_counts_table(df, "label")

    assert set(result.columns) == {"label", "count", "proportion"}
    assert int(result.loc[result["label"] == "a", "count"].iloc[0]) == 2
    assert int(result.loc[result["label"] == "b", "count"].iloc[0]) == 1


def test_grouped_counts_table_returns_group_counts():
    df = pd.DataFrame(
        {
            "original_label": ["Cardboard", "Cardboard", "Textile Trash"],
            "fine_label": ["paper_cardboard", "paper_cardboard", "unknown"],
            "usage": ["known_train", "known_test", "unknown_test"],
        }
    )

    result = grouped_counts_table(df, ["original_label", "fine_label"])

    assert "count" in result.columns
    assert int(
        result.loc[
            result["original_label"] == "Cardboard",
            "count",
        ].iloc[0]
    ) == 2


def test_realwaste_inspection_outputs_exist_after_running():
    assert (
        PROJECT_ROOT / "ml" / "outputs" / "metrics" / "realwaste_inspection_summary_v1.json"
    ).exists()
    assert (
        PROJECT_ROOT / "ml" / "outputs" / "metrics" / "realwaste_fine_label_counts_v1.csv"
    ).exists()
    assert (
        PROJECT_ROOT / "docs" / "results" / "realwaste_inspection_v1_report.md"
    ).exists()


def test_realwaste_inspection_report_mentions_expected_counts():
    text = read_text("docs/results/realwaste_inspection_v1_report.md")

    assert "4752" in text
    assert "4434" in text
    assert "318" in text
    assert "Textile Trash" in text
