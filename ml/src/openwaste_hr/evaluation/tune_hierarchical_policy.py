from __future__ import annotations

import argparse
import itertools
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
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML dictionary.")

    return config


def resolve_path(project_root: str | Path, path_text: str | Path) -> Path:
    return Path(project_root) / Path(path_text)


def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
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


def generate_threshold_grid(search_space: dict[str, list[float]]) -> list[dict[str, float]]:
    required_keys = {
        "fine_confidence_thresholds",
        "coarse_confidence_thresholds",
        "coarse_margin_thresholds",
        "minimum_confidence_for_coarse_values",
    }

    missing_keys = required_keys - set(search_space)

    if missing_keys:
        raise ValueError(f"Search space is missing keys: {sorted(missing_keys)}")

    grid: list[dict[str, float]] = []

    for fine_threshold, coarse_threshold, coarse_margin, minimum_coarse in itertools.product(
        search_space["fine_confidence_thresholds"],
        search_space["coarse_confidence_thresholds"],
        search_space["coarse_margin_thresholds"],
        search_space["minimum_confidence_for_coarse_values"],
    ):
        grid.append(
            {
                "fine_confidence_threshold": float(fine_threshold),
                "coarse_confidence_threshold": float(coarse_threshold),
                "coarse_margin_threshold": float(coarse_margin),
                "minimum_confidence_for_coarse": float(minimum_coarse),
            }
        )

    if not grid:
        raise ValueError("Threshold grid is empty.")

    return grid


def compute_objective_score(
    known_metrics: dict[str, Any],
    unknown_metrics: dict[str, Any],
    objective_weights: dict[str, float],
) -> float:
    unknown_manual_review_rate = float(unknown_metrics["unknown_manual_review_rate"])
    known_success_rate_over_accepted = float(
        known_metrics["hierarchical_success_rate_over_accepted"]
    )
    known_decision_coverage = float(known_metrics["known_decision_coverage"])

    score = (
        unknown_manual_review_rate
        * float(objective_weights["unknown_manual_review_rate"])
        + known_success_rate_over_accepted
        * float(objective_weights["known_success_rate_over_accepted"])
        + known_decision_coverage
        * float(objective_weights["known_decision_coverage"])
    )

    return round(float(score), 6)


def is_policy_valid(
    known_metrics: dict[str, Any],
    selection_config: dict[str, Any],
) -> bool:
    known_decision_coverage = float(known_metrics["known_decision_coverage"])
    known_success_rate_over_accepted = float(
        known_metrics["hierarchical_success_rate_over_accepted"]
    )

    return (
        known_decision_coverage
        >= float(selection_config["minimum_known_decision_coverage"])
        and known_success_rate_over_accepted
        >= float(selection_config["minimum_known_success_rate_over_accepted"])
    )


def evaluate_threshold_candidate(
    threshold_config: dict[str, float],
    known_predictions_df: pd.DataFrame,
    unknown_predictions_df: pd.DataFrame,
    fine_to_coarse: dict[str, str],
    selection_config: dict[str, Any],
) -> dict[str, Any]:
    known_decisions_df = apply_hierarchical_policy(
        predictions_df=known_predictions_df,
        fine_to_coarse=fine_to_coarse,
        fine_confidence_threshold=threshold_config["fine_confidence_threshold"],
        coarse_confidence_threshold=threshold_config["coarse_confidence_threshold"],
        coarse_margin_threshold=threshold_config["coarse_margin_threshold"],
        minimum_confidence_for_coarse=threshold_config["minimum_confidence_for_coarse"],
    )

    unknown_decisions_df = apply_hierarchical_policy(
        predictions_df=unknown_predictions_df,
        fine_to_coarse=fine_to_coarse,
        fine_confidence_threshold=threshold_config["fine_confidence_threshold"],
        coarse_confidence_threshold=threshold_config["coarse_confidence_threshold"],
        coarse_margin_threshold=threshold_config["coarse_margin_threshold"],
        minimum_confidence_for_coarse=threshold_config["minimum_confidence_for_coarse"],
    )

    known_metrics = compute_known_hierarchical_metrics(known_decisions_df)
    unknown_metrics = compute_unknown_hierarchical_metrics(unknown_decisions_df)

    valid_policy = is_policy_valid(
        known_metrics=known_metrics,
        selection_config=selection_config,
    )

    objective_score = compute_objective_score(
        known_metrics=known_metrics,
        unknown_metrics=unknown_metrics,
        objective_weights=selection_config["objective_weights"],
    )

    return {
        **threshold_config,
        "is_valid_policy": bool(valid_policy),
        "objective_score": objective_score,
        **known_metrics,
        **unknown_metrics,
    }


def select_best_policy(sweep_df: pd.DataFrame) -> dict[str, Any]:
    if sweep_df.empty:
        raise ValueError("Cannot select a policy from an empty sweep DataFrame.")

    valid_df = sweep_df[sweep_df["is_valid_policy"] == True].copy()  # noqa: E712

    if valid_df.empty:
        candidate_df = sweep_df.copy()
    else:
        candidate_df = valid_df

    candidate_df = candidate_df.sort_values(
        by=[
            "objective_score",
            "unknown_manual_review_rate",
            "hierarchical_success_rate_over_accepted",
            "known_decision_coverage",
        ],
        ascending=[False, False, False, False],
    )

    return candidate_df.iloc[0].to_dict()


def plot_tuning_results(
    sweep_df: pd.DataFrame,
    selected_policy: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.scatter(
        sweep_df["known_decision_coverage"],
        sweep_df["unknown_manual_review_rate"],
        alpha=0.65,
    )

    axis.scatter(
        [selected_policy["known_decision_coverage"]],
        [selected_policy["unknown_manual_review_rate"]],
        marker="x",
        s=120,
    )

    axis.set_title("Safe Hierarchical Policy Tuning")
    axis.set_xlabel("Known Decision Coverage")
    axis.set_ylabel("Local Unknown Manual Review Rate")
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_markdown_report(
    output_path: Path,
    selected_policy: dict[str, Any],
    top_policies_df: pd.DataFrame,
    decision_distribution_df: pd.DataFrame,
    tuning_plot_relative: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_thresholds_df = pd.DataFrame(
        [
            {
                "threshold": "fine_confidence_threshold",
                "value": selected_policy["fine_confidence_threshold"],
            },
            {
                "threshold": "coarse_confidence_threshold",
                "value": selected_policy["coarse_confidence_threshold"],
            },
            {
                "threshold": "coarse_margin_threshold",
                "value": selected_policy["coarse_margin_threshold"],
            },
            {
                "threshold": "minimum_confidence_for_coarse",
                "value": selected_policy["minimum_confidence_for_coarse"],
            },
        ]
    )

    known_metrics_df = pd.DataFrame(
        [
            {
                "metric": "known_total_samples",
                "value": selected_policy["known_total_samples"],
            },
            {
                "metric": "fine_decision_count",
                "value": selected_policy["fine_decision_count"],
            },
            {
                "metric": "coarse_fallback_count",
                "value": selected_policy["coarse_fallback_count"],
            },
            {
                "metric": "manual_review_count",
                "value": selected_policy["manual_review_count"],
            },
            {
                "metric": "known_decision_coverage",
                "value": selected_policy["known_decision_coverage"],
            },
            {
                "metric": "known_manual_review_rate",
                "value": selected_policy["known_manual_review_rate"],
            },
            {
                "metric": "hierarchical_success_rate_over_all",
                "value": selected_policy["hierarchical_success_rate_over_all"],
            },
            {
                "metric": "hierarchical_success_rate_over_accepted",
                "value": selected_policy["hierarchical_success_rate_over_accepted"],
            },
        ]
    )

    unknown_metrics_df = pd.DataFrame(
        [
            {
                "metric": "unknown_total_samples",
                "value": selected_policy["unknown_total_samples"],
            },
            {
                "metric": "unknown_manual_review_count",
                "value": selected_policy["unknown_manual_review_count"],
            },
            {
                "metric": "unknown_fine_accept_count",
                "value": selected_policy["unknown_fine_accept_count"],
            },
            {
                "metric": "unknown_coarse_accept_count",
                "value": selected_policy["unknown_coarse_accept_count"],
            },
            {
                "metric": "unknown_accepted_count",
                "value": selected_policy["unknown_accepted_count"],
            },
            {
                "metric": "unknown_manual_review_rate",
                "value": selected_policy["unknown_manual_review_rate"],
            },
            {
                "metric": "unknown_acceptance_rate",
                "value": selected_policy["unknown_acceptance_rate"],
            },
        ]
    )

    report = f"""# Safe Hierarchical Policy Tuning v1 Report

## Purpose

This report tunes the OpenWaste-HR hierarchical decision policy to improve local unknown safety while preserving useful known-test decisions.

## Selected Thresholds

{dataframe_to_markdown_table(selected_thresholds_df)}

## Known-Test Metrics

{dataframe_to_markdown_table(known_metrics_df)}

## Local Unknown Metrics

{dataframe_to_markdown_table(unknown_metrics_df)}

## Decision Distribution

{dataframe_to_markdown_table(decision_distribution_df)}

## Top Candidate Policies

{dataframe_to_markdown_table(top_policies_df)}

## Tuning Plot

![Safe hierarchical policy tuning]({tuning_plot_relative})

## Research Interpretation

The tuned policy is selected by balancing known-test usefulness and local unknown safety.

A safer policy should increase the local unknown manual-review rate while avoiding an excessive drop in useful known-test decisions.

This tuning stage shows that hierarchical fallback must be controlled carefully. Coarse fallback is useful for known-class uncertainty, but if it is too permissive it can still accept unknown images.
"""

    output_path.write_text(report, encoding="utf-8")


def run_safe_hierarchical_tuning(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    search_space = config["search_space"]
    selection_config = config["selection"]
    output_config = config["outputs"]

    known_predictions_path = resolve_path(project_root, paths["known_predictions_csv"])
    unknown_predictions_path = resolve_path(project_root, paths["local_unknown_predictions_csv"])

    output_metrics_dir = resolve_path(project_root, paths["output_metrics_dir"])
    output_figures_dir = resolve_path(project_root, paths["output_figures_dir"])
    docs_results_dir = resolve_path(project_root, paths["docs_results_dir"])
    docs_figures_dir = resolve_path(project_root, paths["docs_figures_dir"])

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    docs_results_dir.mkdir(parents=True, exist_ok=True)
    docs_figures_dir.mkdir(parents=True, exist_ok=True)

    known_predictions_df = pd.read_csv(known_predictions_path)
    unknown_predictions_df = pd.read_csv(unknown_predictions_path)

    fine_to_coarse = {
        str(fine_label): str(coarse_label)
        for fine_label, coarse_label in config["labels"]["fine_to_coarse"].items()
    }

    threshold_grid = generate_threshold_grid(search_space)

    sweep_rows = []

    for threshold_config in threshold_grid:
        sweep_rows.append(
            evaluate_threshold_candidate(
                threshold_config=threshold_config,
                known_predictions_df=known_predictions_df,
                unknown_predictions_df=unknown_predictions_df,
                fine_to_coarse=fine_to_coarse,
                selection_config=selection_config,
            )
        )

    sweep_df = pd.DataFrame(sweep_rows)
    selected_policy = select_best_policy(sweep_df)

    selected_threshold_config = {
        "fine_confidence_threshold": float(selected_policy["fine_confidence_threshold"]),
        "coarse_confidence_threshold": float(selected_policy["coarse_confidence_threshold"]),
        "coarse_margin_threshold": float(selected_policy["coarse_margin_threshold"]),
        "minimum_confidence_for_coarse": float(selected_policy["minimum_confidence_for_coarse"]),
    }

    known_decisions_df = apply_hierarchical_policy(
        predictions_df=known_predictions_df,
        fine_to_coarse=fine_to_coarse,
        **selected_threshold_config,
    )

    unknown_decisions_df = apply_hierarchical_policy(
        predictions_df=unknown_predictions_df,
        fine_to_coarse=fine_to_coarse,
        **selected_threshold_config,
    )

    decision_distribution_df = build_hierarchical_decision_distribution(
        known_decisions_df=known_decisions_df,
        unknown_decisions_df=unknown_decisions_df,
    )

    threshold_sweep_path = output_metrics_dir / output_config["threshold_sweep_csv"]
    selected_policy_path = output_metrics_dir / output_config["selected_policy_json"]
    known_decisions_path = output_metrics_dir / output_config["known_decisions_csv"]
    unknown_decisions_path = output_metrics_dir / output_config["local_unknown_decisions_csv"]
    metrics_json_path = output_metrics_dir / output_config["metrics_json"]
    decision_distribution_path = output_metrics_dir / output_config["decision_distribution_csv"]
    tuning_plot_path = output_figures_dir / output_config["tuning_plot"]
    docs_tuning_plot_path = docs_figures_dir / output_config["tuning_plot"]
    markdown_report_path = docs_results_dir / output_config["markdown_report"]

    sweep_df.to_csv(threshold_sweep_path, index=False)
    known_decisions_df.to_csv(known_decisions_path, index=False)
    unknown_decisions_df.to_csv(unknown_decisions_path, index=False)
    decision_distribution_df.to_csv(decision_distribution_path, index=False)

    top_policies_df = sweep_df.sort_values(
        by=[
            "objective_score",
            "unknown_manual_review_rate",
            "hierarchical_success_rate_over_accepted",
            "known_decision_coverage",
        ],
        ascending=[False, False, False, False],
    ).head(10)

    selected_payload = {
        "selected_policy": selected_threshold_config,
        "selection_config": selection_config,
        "selected_metrics": {
            key: selected_policy[key]
            for key in selected_policy
            if key not in selected_threshold_config
        },
    }

    selected_policy_path.write_text(
        json.dumps(selected_payload, indent=2),
        encoding="utf-8",
    )

    metrics_payload = {
        "selected_policy": selected_threshold_config,
        "known_metrics": compute_known_hierarchical_metrics(known_decisions_df),
        "local_unknown_metrics": compute_unknown_hierarchical_metrics(
            unknown_decisions_df
        ),
    }

    metrics_json_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    plot_tuning_results(
        sweep_df=sweep_df,
        selected_policy=selected_policy,
        output_path=tuning_plot_path,
    )

    shutil.copyfile(tuning_plot_path, docs_tuning_plot_path)

    report_columns = [
        "fine_confidence_threshold",
        "coarse_confidence_threshold",
        "coarse_margin_threshold",
        "minimum_confidence_for_coarse",
        "objective_score",
        "known_decision_coverage",
        "hierarchical_success_rate_over_accepted",
        "unknown_manual_review_rate",
        "unknown_acceptance_rate",
    ]

    write_markdown_report(
        output_path=markdown_report_path,
        selected_policy=selected_policy,
        top_policies_df=top_policies_df[report_columns],
        decision_distribution_df=decision_distribution_df,
        tuning_plot_relative=f"figures/{output_config['tuning_plot']}",
    )

    print("Safe hierarchical policy tuning completed successfully.")
    print(f"Threshold candidates evaluated: {len(sweep_df)}")
    print("\nSelected policy:")
    print(json.dumps(selected_threshold_config, indent=2))
    print("\nSelected metrics:")
    print(json.dumps(metrics_payload, indent=2))
    print("\nCreated files:")
    print(f"- threshold sweep: {threshold_sweep_path}")
    print(f"- selected policy: {selected_policy_path}")
    print(f"- known decisions: {known_decisions_path}")
    print(f"- local unknown decisions: {unknown_decisions_path}")
    print(f"- metrics: {metrics_json_path}")
    print(f"- decision distribution: {decision_distribution_path}")
    print(f"- tuning plot: {tuning_plot_path}")
    print(f"- thesis report: {markdown_report_path}")

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune safe hierarchical fine/coarse/manual-review policy."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/safe_hierarchical_policy_tuning.yaml",
        help="Path to safe hierarchical policy tuning YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    run_safe_hierarchical_tuning(
        config_path=args.config,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()