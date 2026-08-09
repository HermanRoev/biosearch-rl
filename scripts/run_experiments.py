#!/usr/bin/env python3
"""Run the complete Milestone 4 matrix and generate its final artifacts."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/biosearch-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/biosearch-cache")

from biosearch.evaluation import (
    MILESTONE4_CONDITIONS,
    ControllerName,
    SensorCondition,
    aggregate_metrics,
    config_for_condition,
)
from biosearch.experiments import (
    experiment_config,
    run_milestone4_experiments,
    save_aggregate_results,
    save_experiment_results,
)
from biosearch.training.evaluate import save_policy_gif
from biosearch.visualization.plots import (
    CONTROLLER_ORDER,
    save_milestone4_metric_plot,
    save_sensor_failure_comparison,
    save_training_curve,
    save_trajectory_comparison,
    save_visitation_heatmap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all Milestone 4 experiments.")
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
    parser.add_argument("--seed-start", type=int, default=3_000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.results_dir / "data"
    figure_dir = args.results_dir / "figures"
    animation_dir = args.results_dir / "animations"
    results = run_milestone4_experiments(
        args.normal_model,
        args.robust_model,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        progress=lambda label: print(f"Evaluating {label}", flush=True),
    )
    csv_path = data_dir / "milestone4_experiments.csv"
    visitation_path = data_dir / "milestone4_visitation.npz"
    save_experiment_results(
        results,
        csv_path=csv_path,
        visitation_path=visitation_path,
    )
    aggregates = aggregate_metrics(results.metrics)
    summary_path = data_dir / "milestone4_summary.csv"
    save_aggregate_results(aggregates, summary_path)
    metadata = {
        "episodes_per_controller_condition": args.episodes,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.episodes - 1,
        "max_steps": args.max_steps,
        "conditions": [condition.value for condition in MILESTONE4_CONDITIONS],
        "controllers": list(CONTROLLER_ORDER),
        "paired_design": (
            "Every controller receives the same start pose, plume seed, "
            "condition configuration, and time limit for a given seed."
        ),
        "normal_model": str(args.normal_model),
        "robust_model": str(args.robust_model),
    }
    metadata_path = data_dir / "milestone4_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_milestone4_metric_plot(
        aggregates,
        figure_dir / "milestone4_success_rate.png",
        metric="success_rate",
    )
    save_milestone4_metric_plot(
        aggregates,
        figure_dir / "milestone4_search_time.png",
        metric="mean_success_steps",
    )
    save_sensor_failure_comparison(
        aggregates,
        figure_dir / "milestone4_sensor_failure.png",
    )
    left_config = config_for_condition(
        SensorCondition.LEFT_DISABLED,
        experiment_config(args.max_steps),
    )
    example_runs = [
        next(
            run
            for run in results.runs
            if run.metrics.controller == controller
            and run.metrics.condition == SensorCondition.LEFT_DISABLED.value
            and run.metrics.seed == args.seed_start
        )
        for controller in CONTROLLER_ORDER
    ]
    save_trajectory_comparison(
        example_runs,
        left_config,
        figure_dir / "milestone4_trajectory_comparison.png",
        title=(
            "Paired trajectories with the left sensor disabled "
            f"(preselected first seed: {args.seed_start})"
        ),
    )
    save_visitation_heatmap(
        results,
        left_config,
        figure_dir / "milestone4_visitation_heatmap.png",
    )
    robust_runs = [
        run
        for run in results.runs
        if run.metrics.controller == ControllerName.PPO_ROBUST.value
        and run.metrics.condition == SensorCondition.LEFT_DISABLED.value
    ]
    gif_run = next((run for run in robust_runs if run.metrics.success), robust_runs[0])
    gif_path = animation_dir / "ppo_robust_left_disabled.gif"
    save_policy_gif(
        args.robust_model,
        seed=gif_run.metrics.seed,
        output_path=gif_path,
        condition=SensorCondition.LEFT_DISABLED,
        controller_name=ControllerName.PPO_ROBUST,
        max_steps=args.max_steps,
    )
    robust_evaluations = Path("logs/ppo_robust/evaluations/evaluations.npz")
    if robust_evaluations.exists():
        save_training_curve(
            robust_evaluations,
            figure_dir / "milestone4_robust_training_curve.png",
        )

    for row in aggregates:
        print(
            f"{row.condition:18s} {row.controller:11s} "
            f"success={row.success_rate:5.1f}% "
            f"steps={row.mean_success_steps:6.1f}",
            flush=True,
        )
    print(f"Saved {len(results.metrics)} episode rows to {csv_path}")
    print(f"Saved aggregate rows to {summary_path}")
    print(f"Saved paired-design metadata to {metadata_path}")
    print(f"Saved robust-policy GIF to {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
