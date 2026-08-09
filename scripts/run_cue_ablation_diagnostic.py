#!/usr/bin/env python3
"""Run the preregistered odor/wind cue-ablation diagnostic."""

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
    ZEROED_WIND_FEATURES,
    AblationEpisode,
    ObservationCondition,
    PairedAblationComparison,
    ablation_episode_from_run,
    aggregate_ablation_episodes,
    apply_observation_condition,
    compare_paired_ablation_outcomes,
    holm_adjusted_p_values,
)
from biosearch.evaluation import ControllerName, SensorCondition
from biosearch.experiments import evaluate_policy_batch, experiment_config
from biosearch.visualization.plots import save_cue_ablation_comparison

INPUT_CONDITIONS = (
    ObservationCondition.UNMASKED,
    ObservationCondition.ODOR_BLIND,
    ObservationCondition.WIND_BLIND,
    ObservationCondition.ODOR_WIND_BLIND,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 5.2 cue ablations.")
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
    parser.add_argument("--seed-start", type=int, default=5_000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser


def _write_dicts(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("episodes and max-steps must be positive")
    for model_path in (args.normal_model, args.robust_model):
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")

    data_dir = args.results_dir / "data"
    figure_dir = args.results_dir / "figures"
    protocol_path = data_dir / "phase5_cue_ablation_protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": "5.2 odor/wind cue ablations",
        "question": "Which observable cue supports odor-independent PPO search?",
        "hypothesis": (
            "Robust PPO learned wind-guided upwind navigation from the consistently downwind start."
        ),
        "physical_condition": SensorCondition.NORMAL.value,
        "policy_input_conditions": [condition.value for condition in INPUT_CONDITIONS],
        "episodes_per_policy_input_condition": args.episodes,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.episodes - 1,
        "max_steps": args.max_steps,
        "paired_design": (
            "All four input conditions use the same pose, simulation seed, physical "
            "plume, and time limit for a given policy and seed."
        ),
        "odor_mask": [
            *ZEROED_ODOR_FEATURES,
            "time_since_detection set to 1.0",
        ],
        "wind_mask": [*ZEROED_WIND_FEATURES, "both components set to 0.0"],
        "predictions_declared_before_evaluation": {
            "robust_wind_shortcut": (
                "wind-hidden robust success drops by at least 30 percentage points "
                "versus full observation with Holm-adjusted p<0.05"
            ),
            "robust_without_both_cues": "combined-mask robust success <=10%",
            "normal_odor_replication": (
                "odor-hidden normal success drops by at least 50 percentage points"
            ),
        },
        "statistics": (
            "95% Wilson intervals; two-sided exact McNemar tests versus full "
            "observation; Holm correction over three tests within each policy"
        ),
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
        for input_condition in INPUT_CONDITIONS:
            print(
                f"Evaluating {controller.value}: {input_condition.value}",
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
                    condition=input_condition,
                ),
            )
            episode_rows.extend(ablation_episode_from_run(run, input_condition) for run in runs)

    summaries = aggregate_ablation_episodes(episode_rows)
    comparisons: list[PairedAblationComparison] = []
    for input_condition in INPUT_CONDITIONS[1:]:
        comparisons.extend(
            compare_paired_ablation_outcomes(
                episode_rows,
                reference_condition=ObservationCondition.UNMASKED,
                comparison_condition=input_condition,
            )
        )
    adjustments = holm_adjusted_p_values(comparisons)
    episode_dicts = [row.as_dict() for row in episode_rows]
    summary_dicts = [row.as_dict() for row in summaries]
    comparison_dicts = [
        {
            **row.as_dict(),
            "holm_adjusted_p_value": adjustments[(row.policy, row.comparison_condition)],
        }
        for row in comparisons
    ]
    _write_dicts(data_dir / "phase5_cue_ablation_episodes.csv", episode_dicts)
    _write_dicts(data_dir / "phase5_cue_ablation_summary.csv", summary_dicts)
    _write_dicts(data_dir / "phase5_cue_ablation_paired.csv", comparison_dicts)

    summary_lookup = {(row.policy, row.observation_condition): row for row in summaries}
    comparison_lookup = {(row.policy, row.comparison_condition): row for row in comparisons}
    robust_policy = ControllerName.PPO_ROBUST.value
    normal_policy = ControllerName.PPO_NORMAL.value
    robust_wind = comparison_lookup[(robust_policy, ObservationCondition.WIND_BLIND.value)]
    robust_both = summary_lookup[(robust_policy, ObservationCondition.ODOR_WIND_BLIND.value)]
    normal_odor = comparison_lookup[(normal_policy, ObservationCondition.ODOR_BLIND.value)]
    analysis = {
        "protocol": str(protocol_path),
        "predictions_supported": {
            "robust_wind_shortcut": (
                robust_wind.success_rate_change_percentage_points <= -30.0
                and adjustments[(robust_policy, ObservationCondition.WIND_BLIND.value)] < 0.05
            ),
            "robust_without_both_cues": robust_both.success_rate <= 10.0,
            "normal_odor_replication": (normal_odor.success_rate_change_percentage_points <= -50.0),
        },
        "paired_comparisons": comparison_dicts,
    }
    analysis_path = data_dir / "phase5_cue_ablation_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    figure_path = figure_dir / "phase5_cue_ablation_success.png"
    save_cue_ablation_comparison(summaries, figure_path)

    for row in summaries:
        print(
            f"{row.policy:11s} {row.observation_condition:15s} "
            f"success={row.success_rate:5.1f}% "
            f"steps={row.mean_success_steps:6.1f}",
            flush=True,
        )
    for row in comparisons:
        print(
            f"{row.policy:11s} full->{row.comparison_condition:15s} "
            f"change={row.success_rate_change_percentage_points:+5.1f} pp "
            f"raw_p={row.exact_mcnemar_p_value:.6g} "
            f"holm_p={adjustments[(row.policy, row.comparison_condition)]:.6g}",
            flush=True,
        )
    print(f"Saved cue-ablation figure to {figure_path}")
    print(f"Saved analysis to {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
