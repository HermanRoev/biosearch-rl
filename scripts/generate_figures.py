#!/usr/bin/env python3
"""Generate Milestone 2 figures and controller animations."""

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

from biosearch.config import SimulationConfig
from biosearch.evaluation import (
    ControllerName,
    EpisodeMetrics,
    SensorCondition,
    aggregate_metrics,
    run_episode,
)
from biosearch.visualization.animation import save_episode_gif
from biosearch.visualization.plots import (
    save_metric_comparison,
    save_training_curve,
    save_trajectory_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Milestone 2 visual results.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/data/milestone2_metrics.csv"),
        help="Episode-level CSV created by evaluate_agents.py.",
    )
    parser.add_argument(
        "--example-seed",
        type=int,
        default=107,
        help="Seed used for the four trajectory examples.",
    )
    parser.add_argument(
        "--skip-gifs",
        action="store_true",
        help="Generate static figures only.",
    )
    parser.add_argument(
        "--training-evaluations",
        type=Path,
        default=Path("logs/ppo/evaluations/evaluations.npz"),
        help="Optional EvalCallback archive used for the PPO learning curve.",
    )
    return parser


def _load_metrics(path: Path) -> list[EpisodeMetrics]:
    if not path.exists():
        raise SystemExit(
            f"Metrics file not found: {path}. Run `python scripts/evaluate_agents.py` first."
        )
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = _load_metrics(args.metrics)
    config = SimulationConfig()

    runs = [
        run_episode(controller, condition, args.example_seed)
        for condition in (SensorCondition.NORMAL, SensorCondition.LEFT_DISABLED)
        for controller in (ControllerName.RANDOM, ControllerName.MOTH)
    ]
    trajectory_path = Path("results/figures/milestone2_trajectories.png")
    metric_path = Path("results/figures/milestone2_metrics.png")
    save_trajectory_comparison(runs, config, trajectory_path)
    save_metric_comparison(aggregate_metrics(metrics), metric_path)
    print(f"Saved {trajectory_path}")
    print(f"Saved {metric_path}")
    if args.training_evaluations.exists():
        training_path = Path("results/figures/milestone3_training.png")
        save_training_curve(args.training_evaluations, training_path)
        print(f"Saved {training_path}")

    if not args.skip_gifs:
        random_path = Path("results/animations/random_normal.gif")
        moth_path = Path("results/animations/moth_normal.gif")
        save_episode_gif(
            ControllerName.RANDOM,
            SensorCondition.NORMAL,
            seed=7,
            output_path=random_path,
            frame_stride=12,
        )
        save_episode_gif(
            ControllerName.MOTH,
            SensorCondition.NORMAL,
            seed=7,
            output_path=moth_path,
            frame_stride=3,
        )
        print(f"Saved {random_path}")
        print(f"Saved {moth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
