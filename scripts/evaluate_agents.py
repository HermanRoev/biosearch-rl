#!/usr/bin/env python3
"""Evaluate random and moth-inspired controllers for Milestone 2."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from biosearch.evaluation import (
    ControllerName,
    EpisodeMetrics,
    SensorCondition,
    aggregate_metrics,
    run_episode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Milestone 2 controllers under normal and failed sensing."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes per controller/condition pair.",
    )
    parser.add_argument("--seed-start", type=int, default=100, help="First episode seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/data/milestone2_metrics.csv"),
        help="Destination CSV path.",
    )
    return parser


def _write_csv(path: Path, metrics: list[EpisodeMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metrics[0].as_dict())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric.as_dict() for metric in metrics)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive.")

    metrics: list[EpisodeMetrics] = []
    controllers = (ControllerName.RANDOM, ControllerName.MOTH)
    conditions = (SensorCondition.NORMAL, SensorCondition.LEFT_DISABLED)
    for condition in conditions:
        for controller in controllers:
            for seed in range(args.seed_start, args.seed_start + args.episodes):
                result = run_episode(controller, condition, seed)
                metrics.append(result.metrics)

    _write_csv(args.output, metrics)
    print(
        f"{'controller':<12} {'condition':<16} {'success':>9} "
        f"{'mean steps*':>12} {'path':>9} {'collisions':>11} {'odor %':>8}"
    )
    for row in aggregate_metrics(metrics):
        print(
            f"{row.controller:<12} {row.condition:<16} "
            f"{row.success_rate:>8.1f}% {row.mean_success_steps:>12.1f} "
            f"{row.mean_path_length:>9.2f} {row.mean_collisions:>11.2f} "
            f"{row.mean_odor_detection_percentage:>8.1f}"
        )
    print("*Mean steps is calculated over successful episodes only.")
    print(f"Saved {len(metrics)} episode rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
