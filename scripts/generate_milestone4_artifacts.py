#!/usr/bin/env python3
"""Regenerate Milestone 4 plots and the robust-policy GIF from saved data."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/biosearch-matplotlib")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/biosearch-cache")

import numpy as np
from stable_baselines3 import PPO

from biosearch.evaluation import (
    ControllerName,
    EpisodeMetrics,
    SensorCondition,
    aggregate_metrics,
    config_for_condition,
    run_episode,
)
from biosearch.experiments import (
    ExperimentResults,
    evaluate_policy_batch,
    experiment_config,
    save_aggregate_results,
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
    parser = argparse.ArgumentParser(description="Generate Milestone 4 artifacts.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/data/milestone4_experiments.csv"),
    )
    parser.add_argument(
        "--visitation",
        type=Path,
        default=Path("results/data/milestone4_visitation.npz"),
    )
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
    parser.add_argument("--example-seed", type=int, default=3_000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--skip-gif", action="store_true")
    return parser


def _load_metrics(path: Path) -> list[EpisodeMetrics]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [
        EpisodeMetrics(
            controller=row["controller"],
            condition=row["condition"],
            seed=int(row["seed"]),
            success=row["success"].lower() == "true",
            steps=int(row["steps"]),
            path_length=float(row["path_length"]),
            collisions=int(row["collisions"]),
            tortuosity=float(row["tortuosity"]),
            odor_detection_percentage=float(row["odor_detection_percentage"]),
            final_distance_to_source=float(row["final_distance_to_source"]),
        )
        for row in rows
    ]


def _load_visitation(path: Path) -> ExperimentResults:
    with np.load(path) as archive:
        x_edges = archive["x_edges"].copy()
        y_edges = archive["y_edges"].copy()
        visitation = {
            tuple(key.split("__", maxsplit=1)): archive[key].copy()
            for key in archive.files
            if "__" in key
        }
    return ExperimentResults(
        runs=(),
        visitation=visitation,
        x_edges=x_edges,
        y_edges=y_edges,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = _load_metrics(args.metrics)
    aggregates = aggregate_metrics(metrics)
    save_aggregate_results(
        aggregates,
        Path("results/data/milestone4_summary.csv"),
    )
    figure_dir = Path("results/figures")
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

    base_config = experiment_config(args.max_steps)
    condition = SensorCondition.LEFT_DISABLED
    condition_config = config_for_condition(condition, base_config)
    example_runs = [
        run_episode(
            controller,
            condition,
            args.example_seed,
            base_config=base_config,
        )
        for controller in (ControllerName.RANDOM, ControllerName.MOTH)
    ]
    for controller, model_path in (
        (ControllerName.PPO_NORMAL, args.normal_model),
        (ControllerName.PPO_ROBUST, args.robust_model),
    ):
        example_runs.extend(
            evaluate_policy_batch(
                PPO.load(model_path, device="cpu"),
                controller_name=controller,
                condition=condition,
                seeds=(args.example_seed,),
                base_config=base_config,
            )
        )
    example_runs.sort(key=lambda run: CONTROLLER_ORDER.index(run.metrics.controller))
    save_trajectory_comparison(
        example_runs,
        condition_config,
        figure_dir / "milestone4_trajectory_comparison.png",
        title=(
            "Paired trajectories with the left sensor disabled "
            f"(preselected first seed: {args.example_seed})"
        ),
    )
    save_visitation_heatmap(
        _load_visitation(args.visitation),
        condition_config,
        figure_dir / "milestone4_visitation_heatmap.png",
    )
    training_archive = Path("logs/ppo_robust/evaluations/evaluations.npz")
    if training_archive.exists():
        save_training_curve(
            training_archive,
            figure_dir / "milestone4_robust_training_curve.png",
        )

    if not args.skip_gif:
        successful_seed = next(
            row.seed
            for row in metrics
            if row.controller == ControllerName.PPO_ROBUST.value
            and row.condition == condition.value
            and row.success
        )
        save_policy_gif(
            args.robust_model,
            seed=successful_seed,
            output_path=Path("results/animations/ppo_robust_left_disabled.gif"),
            condition=condition,
            controller_name=ControllerName.PPO_ROBUST,
            max_steps=args.max_steps,
        )
    suffix = " and robust-policy animation" if not args.skip_gif else ""
    print(f"Regenerated Milestone 4 figures{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
