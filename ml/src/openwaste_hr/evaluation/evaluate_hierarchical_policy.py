from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from openwaste_hr.evaluation.hierarchical_decision import (
    apply_hierarchical_policy,
    build_hierarchical_decision_distribution,
    compute_known_hierarchical_metrics,
    compute_unknown_hierarchical_metrics,
)


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML dictionary.")

    return config


def resolve_path(project_root: str | Path, path_text: str | Path) -> Path:
    """
    Resolve project-relative path.
    """
    return Path(project_root) / Path(path_text)


def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    """
    Convert DataFrame to Markdown without requiring tabulate.
    """
    if dataframe.empty:
        return "_No rows._"

    columns = list(dataframe.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in dataframe.iterrows():
        values = [str(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def plot_decision_distribution(
    distribution_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot decision distribution for known and local unknown datasets.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = distribution_df.pivot_table(
        index="decision_type",
        columns="dataset",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    figure, axis = plt.subplots(figsize=(9, 6))
    pivot.plot(kind="bar", ax=axis)

    axis.set_title("Hierarchical Decision Distribution")
    axis.set_xlabel("Decision Type")
    axis.set_ylabel("Image Count")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_markdown_report(
    output_path: Path,
    policy_config: dict[str, Any],
    known_metrics: dict[str, Any],
    unknown_metrics: dict[str, Any],
    distribution_df: pd.DataFrame,
    decision_distribution_plot_relative: str,
) -> None:
    """
    Write thesis-ready hierarchical policy report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policy_df = pd.DataFrame(
        [
            {
                "threshold": key,
                "value": value,
            }
            for key, value in policy_config.items()
        ]
    )

    known_metrics_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in known_metrics.items()
        ]
    )

    unknown_metrics_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in unknown_metrics.items()
        ]
    )

    report = f"""# Hierarchical Decision Policy v1 Report

## Purpose

This report evaluates the first OpenWaste-HR hierarchical decision policy.

The policy can output:

1. fine label
2. coarse label
3. manual review

## Policy Thresholds

{dataframe_to_markdown_table(policy_df)}

## Known-Test Metrics

{dataframe_to_markdown_table(known_metrics_df)}

## Local Unknown Metrics

{dataframe_to_markdown_table(unknown_metrics_df)}

## Decision Distribution

{dataframe_to_markdown_table(distribution_df)}

## Decision Distribution Plot

![Hierarchical decision distribution]({decision_distribution_plot_relative})

## Research Interpretation

This stage moves OpenWaste-HR beyond simple accept/reject thresholding.

The system now has a middle decision level: coarse fallback. This means it does not have to force a detailed fine label when fine confidence is weak. It can return a broader category when coarse evidence is stable, and it can still send unsafe or ambiguous images to manual review.

This directly supports the OpenWaste-HR goal of safer open-world waste classification.
"""

    output_path.write_text(report, encoding="utf-8")


def run_hierarchical_policy_evaluation(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """
    Run hierarchical decision policy evaluation.
    """
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    labels_config = config["labels"]
    policy_config = config["policy"]
    output_config = config["outputs"]

    known_predictions_path = resolve_path(project_root, paths["known_predictions_csv"])
    local_unknown_predictions_path = resolve_path(
        project_root,
        paths["local_unknown_predictions_csv"],
    )

    output_metrics_dir = resolve_path(project_root, paths["output_metrics_dir"])
    output_figures_dir = resolve_path(project_root, paths["output_figures_dir"])
    docs_results_dir = resolve_path(project_root, paths["docs_results_dir"])
    docs_figures_dir = resolve_path(project_root, paths["docs_figures_dir"])

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    docs_results_dir.mkdir(parents=True, exist_ok=True)
    docs_figures_dir.mkdir(parents=True, exist_ok=True)

    if not known_predictions_path.exists():
        raise FileNotFoundError(
            f"Known prediction file not found: {known_predictions_path}"
        )

    if not local_unknown_predictions_path.exists():
        raise FileNotFoundError(
            f"Local unknown prediction file not found: {local_unknown_predictions_path}"
        )

    known_predictions_df = pd.read_csv(known_predictions_path)
    unknown_predictions_df = pd.read_csv(local_unknown_predictions_path)

    fine_to_coarse = {
        str(fine_label): str(coarse_label)
        for fine_label, coarse_label in labels_config["fine_to_coarse"].items()
    }

    known_decisions_df = apply_hierarchical_policy(
        predictions_df=known_predictions_df,
        fine_to_coarse=fine_to_coarse,
        fine_confidence_threshold=float(policy_config["fine_confidence_threshold"]),
        coarse_confidence_threshold=float(policy_config["coarse_confidence_threshold"]),
        coarse_margin_threshold=float(policy_config["coarse_margin_threshold"]),
        minimum_confidence_for_coarse=float(
            policy_config["minimum_confidence_for_coarse"]
        ),
    )

    unknown_decisions_df = apply_hierarchical_policy(
        predictions_df=unknown_predictions_df,
        fine_to_coarse=fine_to_coarse,
        fine_confidence_threshold=float(policy_config["fine_confidence_threshold"]),
        coarse_confidence_threshold=float(policy_config["coarse_confidence_threshold"]),
        coarse_margin_threshold=float(policy_config["coarse_margin_threshold"]),
        minimum_confidence_for_coarse=float(
            policy_config["minimum_confidence_for_coarse"]
        ),
    )

    known_metrics = compute_known_hierarchical_metrics(known_decisions_df)
    unknown_metrics = compute_unknown_hierarchical_metrics(unknown_decisions_df)

    distribution_df = build_hierarchical_decision_distribution(
        known_decisions_df=known_decisions_df,
        unknown_decisions_df=unknown_decisions_df,
    )

    known_decisions_path = output_metrics_dir / output_config["known_decisions_csv"]
    unknown_decisions_path = output_metrics_dir / output_config["unknown_decisions_csv"]
    metrics_json_path = output_metrics_dir / output_config["metrics_json"]
    decision_distribution_path = (
        output_metrics_dir / output_config["decision_distribution_csv"]
    )

    known_decisions_df.to_csv(known_decisions_path, index=False)
    unknown_decisions_df.to_csv(unknown_decisions_path, index=False)
    distribution_df.to_csv(decision_distribution_path, index=False)

    metrics_payload = {
        "policy": policy_config,
        "known_metrics": known_metrics,
        "local_unknown_metrics": unknown_metrics,
    }

    metrics_json_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    decision_distribution_plot_path = (
        output_figures_dir / output_config["decision_distribution_plot"]
    )

    plot_decision_distribution(
        distribution_df=distribution_df,
        output_path=decision_distribution_plot_path,
    )

    docs_decision_distribution_plot_path = (
        docs_figures_dir / output_config["decision_distribution_plot"]
    )

    shutil.copyfile(
        decision_distribution_plot_path,
        docs_decision_distribution_plot_path,
    )

    markdown_report_path = docs_results_dir / output_config["markdown_report"]

    write_markdown_report(
        output_path=markdown_report_path,
        policy_config=policy_config,
        known_metrics=known_metrics,
        unknown_metrics=unknown_metrics,
        distribution_df=distribution_df,
        decision_distribution_plot_relative=(
            f"figures/{output_config['decision_distribution_plot']}"
        ),
    )

    print("Hierarchical decision policy evaluation completed successfully.")
    print("\nPolicy:")
    print(json.dumps(policy_config, indent=2))
    print("\nKnown-test metrics:")
    print(json.dumps(known_metrics, indent=2))
    print("\nLocal unknown metrics:")
    print(json.dumps(unknown_metrics, indent=2))
    print("\nCreated files:")
    print(f"- known decisions: {known_decisions_path}")
    print(f"- local unknown decisions: {unknown_decisions_path}")
    print(f"- metrics: {metrics_json_path}")
    print(f"- decision distribution: {decision_distribution_path}")
    print(f"- decision plot: {decision_distribution_plot_path}")
    print(f"- thesis report: {markdown_report_path}")

    return {
        "known_metrics": known_metrics,
        "local_unknown_metrics": unknown_metrics,
        "markdown_report": str(markdown_report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical fine/coarse/manual-review decision policy."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/hierarchical_decision_policy.yaml",
        help="Path to hierarchical decision policy YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    run_hierarchical_policy_evaluation(
        config_path=args.config,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()