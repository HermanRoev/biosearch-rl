#!/usr/bin/env python3
"""Evaluate a saved PPO model and animate its first successful episode."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from biosearch.evaluation import SensorCondition
from biosearch.training.evaluate import (
    PolicyEvaluation,
    evaluate_policy_episodes,
    save_policy_gif,
    summarize_policy_evaluations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the Milestone 3 PPO policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best/best_model.zip"),
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=2_000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--condition",
        choices=tuple(condition.value for condition in SensorCondition),
        default=SensorCondition.NORMAL.value,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/data/milestone3_rl_evaluation.csv"),
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=Path("results/animations/ppo_normal.gif"),
    )
    return parser


def _write_csv(path: Path, evaluations: list[PolicyEvaluation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*evaluations[0].metrics.as_dict(), "total_reward"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(
                {
                    **evaluation.metrics.as_dict(),
                    "total_reward": evaluation.total_reward,
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    condition = SensorCondition(args.condition)
    evaluations = evaluate_policy_episodes(
        args.model,
        episodes=args.episodes,
        seed_start=args.seed_start,
        condition=condition,
        max_steps=args.max_steps,
    )
    _write_csv(args.output, evaluations)
    summary = summarize_policy_evaluations(evaluations)
    print(
        f"condition={condition.value} episodes={len(evaluations)} "
        f"success_rate={summary['success_rate']:.1f}% "
        f"mean_success_steps={summary['mean_success_steps']:.1f} "
        f"mean_reward={summary['mean_reward']:.2f}"
    )
    print(f"Saved evaluation rows to {args.output}")

    successful = next(
        (evaluation for evaluation in evaluations if evaluation.metrics.success),
        None,
    )
    if successful is None:
        print("No successful evaluation episode was available for GIF generation.")
        return 2
    gif_metrics = save_policy_gif(
        args.model,
        seed=successful.metrics.seed,
        output_path=args.gif,
        condition=condition,
        max_steps=args.max_steps,
    )
    print(
        f"Saved successful policy GIF to {args.gif} "
        f"(seed={gif_metrics.seed}, steps={gif_metrics.steps})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
