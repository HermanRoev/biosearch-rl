#!/usr/bin/env python3
"""Run the preregistered inference-only odor-blind policy diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/biosearch-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/biosearch-cache")

from stable_baselines3 import PPO

from biosearch.ablations import (
    ZEROED_ODOR_FEATURES,
    AblationEpisode,
    AblationSummary,
    ObservationCondition,
    PairedAblationComparison,
    ablation_episode_from_run,
    aggregate_ablation_episodes,
    apply_observation_condition,
    compare_paired_ablation_outcomes,
)
from biosearch.evaluation import ControllerName, SensorCondition
from biosearch.experiments import evaluate_policy_batch, experiment_config
from biosearch.visualization.plots import save_odor_blind_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 5 odor-blind diagnostic.")
    parser.add_argument(
        "--normal-model",
        type=Path,
        default=Path("models/best/best_model.zip"),
    )
    parser.add_argument(
        "--robust-model",
        type=Path,
        default=Path("models/robust/best/best_model.zip"),
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=4_000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser


def _write_rows(path: Path, rows: Sequence[object]) -> None:
    dictionaries = [row.as_dict() for row in rows]  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def _interpret_result(
    summary: AblationSummary,
    comparison: PairedAblationComparison,
) -> str:
    if summary.success_rate <= 10.0 and comparison.success_rate_change_percentage_points <= -50.0:
        return "strong_evidence_of_odor_dependence"
    if summary.success_rate >= 50.0:
        return "substantial_odor_independent_shortcut_performance"
    return "mixed_or_partial_odor_dependence"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("episodes and max-steps must be positive")
    for model_path in (args.normal_model, args.robust_model):
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")

    data_dir = args.results_dir / "data"
    figure_dir = args.results_dir / "figures"
    protocol_path = data_dir / "phase5_odor_blind_protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": "5.1 odor-blind diagnostic",
        "question": "Can PPO find the source when odor-derived policy inputs are hidden?",
        "physical_condition": SensorCondition.NORMAL.value,
        "episodes_per_policy_input_condition": args.episodes,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.episodes - 1,
        "max_steps": args.max_steps,
        "paired_design": (
            "For each policy and seed, unmasked and odor-blind episodes use the "
            "same start pose, simulation seed, physical plume, and time limit."
        ),
        "masked_policy_features": [
            *ZEROED_ODOR_FEATURES,
            "time_since_detection set to 1.0 (permanent absence)",
        ],
        "preserved_policy_features": [
            "wind_heading_sin",
            "wind_heading_cos",
            "previous_action",
            "front_obstacle_proximity",
            "left_obstacle_proximity",
            "right_obstacle_proximity",
        ],
        "preserved_simulator_state": [
            "physical plume",
            "physical sensor readings",
            "recorded odor detections",
            "source and obstacles",
        ],
        "interpretation_thresholds_declared_before_evaluation": {
            "strong_evidence_of_odor_dependence": (
                "odor-blind success <=10% and paired change <=-50 percentage points"
            ),
            "substantial_shortcut_performance": "odor-blind success >=50%",
            "otherwise": "mixed or partial odor dependence",
        },
        "primary_statistic": "paired success-rate change",
        "paired_test": "two-sided exact McNemar test",
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote fixed protocol to {protocol_path}", flush=True)

    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    base_config = experiment_config(args.max_steps)
    models = {
        ControllerName.PPO_NORMAL: PPO.load(args.normal_model, device="cpu"),
        ControllerName.PPO_ROBUST: PPO.load(args.robust_model, device="cpu"),
    }
    episode_rows: list[AblationEpisode] = []
    for controller, model in models.items():
        for observation_condition in (
            ObservationCondition.UNMASKED,
            ObservationCondition.ODOR_BLIND,
        ):
            print(
                f"Evaluating {controller.value}: {observation_condition.value}",
                flush=True,
            )
            runs = evaluate_policy_batch(
                model,
                controller_name=controller,
                condition=SensorCondition.NORMAL,
                seeds=seeds,
                base_config=base_config,
                observation_transform=partial(
                    apply_observation_condition,
                    condition=observation_condition,
                ),
            )
            episode_rows.extend(
                ablation_episode_from_run(run, observation_condition) for run in runs
            )

    summaries = aggregate_ablation_episodes(episode_rows)
    comparisons = compare_paired_ablation_outcomes(episode_rows)
    episode_path = data_dir / "phase5_odor_blind_episodes.csv"
    summary_path = data_dir / "phase5_odor_blind_summary.csv"
    paired_path = data_dir / "phase5_odor_blind_paired.csv"
    _write_rows(episode_path, episode_rows)
    _write_rows(summary_path, summaries)
    _write_rows(paired_path, comparisons)

    blind_summaries = {
        row.policy: row
        for row in summaries
        if row.observation_condition == ObservationCondition.ODOR_BLIND.value
    }
    analysis = {
        "protocol": str(protocol_path),
        "classifications": {
            comparison.policy: _interpret_result(
                blind_summaries[comparison.policy],
                comparison,
            )
            for comparison in comparisons
        },
        "paired_comparisons": [row.as_dict() for row in comparisons],
    }
    analysis_path = data_dir / "phase5_odor_blind_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    figure_path = figure_dir / "phase5_odor_blind_success.png"
    save_odor_blind_comparison(summaries, comparisons, figure_path)

    for row in summaries:
        print(
            f"{row.policy:11s} {row.observation_condition:10s} "
            f"success={row.success_rate:5.1f}% "
            f"steps={row.mean_success_steps:6.1f} "
            f"actual_odor={row.mean_actual_odor_detection_percentage:5.1f}%",
            flush=True,
        )
    for comparison in comparisons:
        print(
            f"{comparison.policy}: paired change="
            f"{comparison.success_rate_change_percentage_points:+.1f} pp, "
            f"unmasked-only={comparison.reference_only_succeeds}, "
            f"blind-only={comparison.comparison_only_succeeds}, "
            f"exact p={comparison.exact_mcnemar_p_value:.6g}",
            flush=True,
        )
    print(f"Saved diagnostic figure to {figure_path}")
    print(f"Saved analysis to {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
