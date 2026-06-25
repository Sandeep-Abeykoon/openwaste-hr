from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


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


def get_probability_columns(dataframe: pd.DataFrame) -> list[str]:
    probability_columns = [
        column
        for column in dataframe.columns
        if str(column).startswith("prob_")
    ]

    if not probability_columns:
        raise ValueError("No probability columns found. Expected columns like prob_plastic.")

    return probability_columns


def compute_normalized_entropy(probabilities: list[float]) -> float:
    clean_probabilities = [
        max(float(probability), 0.0)
        for probability in probabilities
    ]

    total_probability = sum(clean_probabilities)

    if total_probability <= 0:
        return 0.0

    normalized_probabilities = [
        probability / total_probability
        for probability in clean_probabilities
    ]

    entropy = 0.0

    for probability in normalized_probabilities:
        if probability > 0:
            entropy -= probability * math.log(probability)

    max_entropy = math.log(len(normalized_probabilities))

    if max_entropy <= 0:
        return 0.0

    return round(float(entropy / max_entropy), 6)


def validate_required_columns(dataframe: pd.DataFrame) -> bool:
    required_columns = {
        "sample_id",
        "image_path",
        "pred_label",
        "max_softmax_confidence",
        "hierarchical_decision_type",
        "hierarchical_final_label",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Decision DataFrame is missing required columns: {sorted(missing_columns)}"
        )

    return True


def get_active_learning_reason(decision_type: str) -> str:
    if decision_type == "manual_review":
        return "manual_review_candidate"

    if decision_type == "coarse_label":
        return "coarse_fallback_unknown_candidate"

    if decision_type == "fine_label":
        return "fine_accepted_unknown_candidate"

    return "unknown_candidate"


def get_recommended_human_action(decision_type: str) -> str:
    if decision_type == "manual_review":
        return "Assign true label or confirm unknown/manual-review status."

    if decision_type == "coarse_label":
        return "Check whether coarse fallback is valid or whether the item should remain unknown."

    if decision_type == "fine_label":
        return "Inspect high-confidence known-label acceptance on a local unknown image."

    return "Review sample and assign annotation decision."


def add_active_learning_scores(
    decisions_df: pd.DataFrame,
    decision_priority_weights: dict[str, float],
    score_weights: dict[str, float],
) -> pd.DataFrame:
    validate_required_columns(decisions_df)

    probability_columns = get_probability_columns(decisions_df)

    result = decisions_df.copy()

    entropy_scores: list[float] = []
    confidence_uncertainties: list[float] = []
    coarse_margin_uncertainties: list[float] = []
    decision_priority_scores: list[float] = []
    active_learning_scores: list[float] = []
    active_learning_reasons: list[str] = []
    recommended_actions: list[str] = []

    for _, row in result.iterrows():
        decision_type = str(row["hierarchical_decision_type"])
        confidence = float(row["max_softmax_confidence"])

        probabilities = [
            float(row[column])
            for column in probability_columns
        ]

        entropy_score = compute_normalized_entropy(probabilities)
        confidence_uncertainty = round(float(1.0 - confidence), 6)

        coarse_margin = float(row["coarse_margin"]) if "coarse_margin" in row else 0.0
        bounded_margin = min(max(coarse_margin, 0.0), 1.0)
        coarse_margin_uncertainty = round(float(1.0 - bounded_margin), 6)

        decision_priority = float(
            decision_priority_weights.get(decision_type, 0.50)
        )

        active_learning_score = (
            float(score_weights["decision_priority"]) * decision_priority
            + float(score_weights["entropy"]) * entropy_score
            + float(score_weights["confidence_uncertainty"]) * confidence_uncertainty
            + float(score_weights["coarse_margin_uncertainty"])
            * coarse_margin_uncertainty
        )

        entropy_scores.append(entropy_score)
        confidence_uncertainties.append(confidence_uncertainty)
        coarse_margin_uncertainties.append(coarse_margin_uncertainty)
        decision_priority_scores.append(round(float(decision_priority), 6))
        active_learning_scores.append(round(float(active_learning_score), 6))
        active_learning_reasons.append(get_active_learning_reason(decision_type))
        recommended_actions.append(get_recommended_human_action(decision_type))

    result["entropy_score"] = entropy_scores
    result["confidence_uncertainty"] = confidence_uncertainties
    result["coarse_margin_uncertainty"] = coarse_margin_uncertainties
    result["decision_priority_score"] = decision_priority_scores
    result["active_learning_score"] = active_learning_scores
    result["active_learning_reason"] = active_learning_reasons
    result["recommended_human_action"] = recommended_actions
    result["proposed_annotation_status"] = "needs_human_label"

    return result


def select_candidates_by_quota(
    scored_df: pd.DataFrame,
    total_candidates: int,
    quota_by_decision_type: dict[str, int],
) -> pd.DataFrame:
    """
    Select active learning candidates using decision-type quotas.

    The final number of selected rows must never exceed total_candidates,
    even when the quota values add up to more than total_candidates.
    """
    if total_candidates <= 0:
        raise ValueError("total_candidates must be greater than zero.")

    if scored_df.empty:
        raise ValueError("Cannot select candidates from an empty DataFrame.")

    max_candidates = min(total_candidates, len(scored_df))
    selected_indices: list[int] = []

    priority_order = ["manual_review", "coarse_label", "fine_label"]

    for decision_type in priority_order:
        remaining_slots = max_candidates - len(selected_indices)

        if remaining_slots <= 0:
            break

        quota = int(quota_by_decision_type.get(decision_type, 0))

        if quota <= 0:
            continue

        allowed_count = min(quota, remaining_slots)

        decision_df = scored_df[
            scored_df["hierarchical_decision_type"] == decision_type
        ].copy()

        decision_df = decision_df.sort_values(
            by=["active_learning_score", "sample_id"],
            ascending=[False, True],
        )

        selected_indices.extend(list(decision_df.head(allowed_count).index))

    selected_indices = list(dict.fromkeys(selected_indices))

    remaining_slots = max_candidates - len(selected_indices)

    if remaining_slots > 0:
        remaining_df = scored_df.drop(index=selected_indices, errors="ignore").copy()

        remaining_df = remaining_df.sort_values(
            by=["active_learning_score", "sample_id"],
            ascending=[False, True],
        )

        selected_indices.extend(list(remaining_df.head(remaining_slots).index))

    selected_indices = selected_indices[:max_candidates]

    selected_df = scored_df.loc[selected_indices].copy()

    selected_df = selected_df.sort_values(
        by=["active_learning_score", "sample_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    selected_df.insert(0, "candidate_rank", range(1, len(selected_df) + 1))

    return selected_df

def build_candidate_distribution(candidates_df: pd.DataFrame) -> pd.DataFrame:
    total_candidates = len(candidates_df)

    rows: list[dict[str, Any]] = []

    for decision_type in ["manual_review", "coarse_label", "fine_label"]:
        count = int(
            (candidates_df["hierarchical_decision_type"] == decision_type).sum()
        )

        rows.append(
            {
                "decision_type": decision_type,
                "candidate_count": count,
                "percentage": round(float(count / total_candidates * 100), 2)
                if total_candidates > 0
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def summarize_candidates(candidates_df: pd.DataFrame) -> dict[str, Any]:
    distribution_df = build_candidate_distribution(candidates_df)

    return {
        "selected_candidate_count": int(len(candidates_df)),
        "manual_review_candidates": int(
            distribution_df.loc[
                distribution_df["decision_type"] == "manual_review",
                "candidate_count",
            ].iloc[0]
        ),
        "coarse_label_candidates": int(
            distribution_df.loc[
                distribution_df["decision_type"] == "coarse_label",
                "candidate_count",
            ].iloc[0]
        ),
        "fine_label_candidates": int(
            distribution_df.loc[
                distribution_df["decision_type"] == "fine_label",
                "candidate_count",
            ].iloc[0]
        ),
        "mean_active_learning_score": round(
            float(candidates_df["active_learning_score"].mean()),
            6,
        ),
        "max_active_learning_score": round(
            float(candidates_df["active_learning_score"].max()),
            6,
        ),
        "min_active_learning_score": round(
            float(candidates_df["active_learning_score"].min()),
            6,
        ),
    }


def plot_candidate_scores(
    candidates_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6))

    axis.bar(
        candidates_df["sample_id"],
        candidates_df["active_learning_score"],
    )

    axis.set_title("Active Learning Candidate Scores")
    axis.set_xlabel("Sample ID")
    axis.set_ylabel("Active Learning Score")
    axis.tick_params(axis="x", rotation=70)
    axis.grid(True, axis="y")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def merge_collection_sheet(
    decisions_df: pd.DataFrame,
    collection_sheet_df: pd.DataFrame,
) -> pd.DataFrame:
    collection_columns = [
        "sample_id",
        "object_description",
        "why_unknown_or_difficult",
        "lighting_condition",
        "background_condition",
        "human_note",
        "usage",
    ]

    available_columns = [
        column
        for column in collection_columns
        if column in collection_sheet_df.columns
    ]

    if "sample_id" not in available_columns:
        return decisions_df

    collection_subset = collection_sheet_df[available_columns].copy()

    merged_df = decisions_df.merge(
        collection_subset,
        on="sample_id",
        how="left",
    )

    return merged_df


def write_markdown_report(
    output_path: Path,
    summary: dict[str, Any],
    distribution_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    candidate_score_plot_relative: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in summary.items()
        ]
    )

    candidate_columns = [
        "candidate_rank",
        "sample_id",
        "object_description",
        "hierarchical_decision_type",
        "hierarchical_final_label",
        "pred_label",
        "max_softmax_confidence",
        "active_learning_score",
        "active_learning_reason",
    ]

    available_candidate_columns = [
        column
        for column in candidate_columns
        if column in candidates_df.columns
    ]

    report_candidates_df = candidates_df[available_candidate_columns].copy()

    report = f"""# Active Learning Candidate Selection v1 Report

## Purpose

This report lists the local unknown dataset images selected for human labelling.

The selection prioritises samples that are useful for improving the OpenWaste-HR uncertainty and manual-review pipeline.

## Candidate Summary

{dataframe_to_markdown_table(summary_df)}

## Candidate Distribution

{dataframe_to_markdown_table(distribution_df)}

## Selected Candidates

{dataframe_to_markdown_table(report_candidates_df)}

## Candidate Score Plot

![Active learning candidate scores]({candidate_score_plot_relative})

## Research Interpretation

The selected candidates form the next human-labelling batch.

Manual-review candidates help confirm genuinely uncertain samples. Coarse-label candidates help check whether broad fallback is safe. Fine-label candidates help identify cases where the model confidently accepted a local unknown image as a known label.

This supports the OpenWaste-HR active learning loop by turning uncertain and risky local cases into labelled feedback for later improvement.
"""

    output_path.write_text(report, encoding="utf-8")


def run_active_learning_selection(
    config_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    project_root = Path(project_root)
    config = load_yaml(config_path)

    paths = config["paths"]
    selection_config = config["selection"]
    outputs = config["outputs"]

    decisions_path = resolve_path(project_root, paths["local_unknown_decisions_csv"])
    collection_sheet_path = resolve_path(
        project_root,
        paths["local_unknown_collection_sheet_csv"],
    )

    output_metrics_dir = resolve_path(project_root, paths["output_metrics_dir"])
    output_figures_dir = resolve_path(project_root, paths["output_figures_dir"])
    docs_results_dir = resolve_path(project_root, paths["docs_results_dir"])
    docs_figures_dir = resolve_path(project_root, paths["docs_figures_dir"])

    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    docs_results_dir.mkdir(parents=True, exist_ok=True)
    docs_figures_dir.mkdir(parents=True, exist_ok=True)

    if not decisions_path.exists():
        raise FileNotFoundError(f"Decision file not found: {decisions_path}")

    if not collection_sheet_path.exists():
        raise FileNotFoundError(f"Collection sheet not found: {collection_sheet_path}")

    decisions_df = pd.read_csv(decisions_path)
    collection_sheet_df = pd.read_csv(collection_sheet_path)

    merged_df = merge_collection_sheet(
        decisions_df=decisions_df,
        collection_sheet_df=collection_sheet_df,
    )

    scored_df = add_active_learning_scores(
        decisions_df=merged_df,
        decision_priority_weights=selection_config["decision_priority_weights"],
        score_weights=selection_config["score_weights"],
    )

    candidates_df = select_candidates_by_quota(
        scored_df=scored_df,
        total_candidates=int(selection_config["total_candidates"]),
        quota_by_decision_type=selection_config["quota_by_decision_type"],
    )

    distribution_df = build_candidate_distribution(candidates_df)
    summary = summarize_candidates(candidates_df)

    candidate_columns = [
        "candidate_rank",
        "sample_id",
        "image_path",
        "object_description",
        "why_unknown_or_difficult",
        "lighting_condition",
        "background_condition",
        "pred_label",
        "max_softmax_confidence",
        "top_coarse_label",
        "top_coarse_confidence",
        "coarse_margin",
        "hierarchical_decision_type",
        "hierarchical_final_label",
        "hierarchical_decision_reason",
        "entropy_score",
        "confidence_uncertainty",
        "coarse_margin_uncertainty",
        "decision_priority_score",
        "active_learning_score",
        "active_learning_reason",
        "recommended_human_action",
        "proposed_annotation_status",
    ]

    available_candidate_columns = [
        column
        for column in candidate_columns
        if column in candidates_df.columns
    ]

    candidates_output_df = candidates_df[available_candidate_columns].copy()

    candidates_path = output_metrics_dir / outputs["candidates_csv"]
    summary_path = output_metrics_dir / outputs["summary_json"]
    distribution_path = output_metrics_dir / outputs["decision_distribution_csv"]
    candidate_score_plot_path = output_figures_dir / outputs["candidate_score_plot"]
    docs_candidate_score_plot_path = docs_figures_dir / outputs["candidate_score_plot"]
    markdown_report_path = docs_results_dir / outputs["markdown_report"]

    candidates_output_df.to_csv(candidates_path, index=False)
    distribution_df.to_csv(distribution_path, index=False)

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_candidate_scores(
        candidates_df=candidates_output_df,
        output_path=candidate_score_plot_path,
    )

    shutil.copyfile(
        candidate_score_plot_path,
        docs_candidate_score_plot_path,
    )

    write_markdown_report(
        output_path=markdown_report_path,
        summary=summary,
        distribution_df=distribution_df,
        candidates_df=candidates_output_df,
        candidate_score_plot_relative=f"figures/{outputs['candidate_score_plot']}",
    )

    print("Active learning candidate selection completed successfully.")
    print(f"Total local unknown samples: {len(decisions_df)}")
    print(f"Selected candidates: {len(candidates_output_df)}")
    print("\nCandidate summary:")
    print(json.dumps(summary, indent=2))
    print("\nCreated files:")
    print(f"- candidates: {candidates_path}")
    print(f"- summary: {summary_path}")
    print(f"- distribution: {distribution_path}")
    print(f"- score plot: {candidate_score_plot_path}")
    print(f"- thesis report: {markdown_report_path}")

    return {
        "summary": summary,
        "candidates_csv": str(candidates_path),
        "markdown_report": str(markdown_report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select active learning candidates from local unknown decisions."
    )
    parser.add_argument(
        "--config",
        default="ml/configs/active_learning_selection.yaml",
        help="Path to active learning selection YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )

    args = parser.parse_args()

    run_active_learning_selection(
        config_path=args.config,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()