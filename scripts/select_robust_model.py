#!/usr/bin/env python3
"""Select a robust PPO checkpoint on validation seeds, not final test seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from stable_baselines3 import PPO

from biosearch.evaluation import MILESTONE4_CONDITIONS, ControllerName
from biosearch.experiments import evaluate_policy_batch, experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate robust PPO checkpoints.")
    parser.add_argument(
        "candidates",
        nargs="*",
        type=Path,
        default=[
            Path("models/robust/best/best_model.zip"),
            Path("models/robust/ppo_biosearch_final.zip"),
        ],
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=2_500)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/data/robust_checkpoint_validation.csv"),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("results/data/robust_checkpoint_selection.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    base_config = experiment_config(args.max_steps)
    rows: list[dict[str, object]] = []
    rankings: list[tuple[float, float, Path]] = []
    for candidate in args.candidates:
        if not candidate.exists():
            raise SystemExit(f"Checkpoint not found: {candidate}")
        model = PPO.load(candidate, device="cpu")
        candidate_successes: list[bool] = []
        candidate_success_steps: list[int] = []
        for condition in MILESTONE4_CONDITIONS:
            runs = evaluate_policy_batch(
                model,
                controller_name=ControllerName.PPO_ROBUST,
                condition=condition,
                seeds=seeds,
                base_config=base_config,
            )
            successes = [run.metrics.success for run in runs]
            successful_steps = [run.metrics.steps for run in runs if run.metrics.success]
            candidate_successes.extend(successes)
            candidate_success_steps.extend(successful_steps)
            rows.append(
                {
                    "checkpoint": str(candidate),
                    "condition": condition.value,
                    "episodes": len(runs),
                    "success_rate": 100.0 * fmean(successes),
                    "mean_success_steps": (
                        fmean(successful_steps) if successful_steps else float("nan")
                    ),
                }
            )
        overall_success = 100.0 * fmean(candidate_successes)
        overall_steps = fmean(candidate_success_steps) if candidate_success_steps else float("inf")
        rankings.append((overall_success, -overall_steps, candidate))
        print(
            f"{candidate}: validation success={overall_success:.1f}% "
            f"mean_success_steps={overall_steps:.1f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overall_success, negative_steps, selected = max(rankings)
    selection = {
        "selected_checkpoint": str(selected),
        "selection_rule": (
            "Highest pooled validation success across six fixed conditions; "
            "mean successful steps breaks ties."
        ),
        "validation_seed_start": args.seed_start,
        "validation_seed_end": args.seed_start + args.episodes - 1,
        "episodes_per_condition": args.episodes,
        "pooled_success_rate": overall_success,
        "mean_success_steps": -negative_steps,
    }
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    print(f"Selected {selected}")
    print(f"Saved selection record to {args.selection_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
