#!/usr/bin/env python3
"""Learn search behavior and an AIRL reward from synthetic moth demonstrations."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean

import matplotlib
import numpy as np
import torch
from stable_baselines3 import PPO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biosearch.config import SimulationConfig
from biosearch.evaluation import ControllerName, SensorCondition, run_episode
from biosearch.irl import (
    AIRLRewardNetwork,
    PolicyEpisode,
    action_accuracy,
    attach_airl_reward_environment,
    behavior_clone,
    collect_moth_demonstrations,
    collect_policy_transitions,
    compare_paired_success,
    evaluate_policy,
    make_ppo,
    summarize_policy,
    train_airl_discriminator,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demonstrations", type=int, default=100)
    parser.add_argument("--demo-seed-start", type=int, default=8_000)
    parser.add_argument("--held-out-demo-seed-start", type=int, default=11_000)
    parser.add_argument("--validation-seed-start", type=int, default=9_000)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--evaluation-seed-start", type=int, default=15_000)
    parser.add_argument("--evaluation-episodes", type=int, default=50)
    parser.add_argument("--airl-rounds", type=int, default=6)
    parser.add_argument("--round-timesteps", type=int, default=4_096)
    parser.add_argument("--generator-samples", type=int, default=8_192)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--models-dir", type=Path, default=Path("models/irl"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--normal-model", type=Path, default=Path("models/best/best_model.zip"))
    parser.add_argument(
        "--robust-model",
        type=Path,
        default=Path("models/robust/best/best_model.zip"),
    )
    return parser


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selection_key(rows: list[PolicyEpisode]) -> tuple[float, float]:
    success_rate = sum(row.success for row in rows) / len(rows)
    successful_steps = [row.steps for row in rows if row.success]
    mean_steps = fmean(successful_steps) if successful_steps else float("inf")
    return success_rate, -mean_steps


def _moth_rows(seeds: range, max_steps: int) -> list[PolicyEpisode]:
    rows: list[PolicyEpisode] = []
    base = SimulationConfig()
    config = replace(base, agent=replace(base.agent, max_steps=max_steps))
    for seed in seeds:
        run = run_episode(
            ControllerName.MOTH,
            SensorCondition.NORMAL,
            seed,
            base_config=config,
        )
        rows.append(
            PolicyEpisode(
                policy="moth",
                seed=seed,
                success=run.metrics.success,
                steps=run.metrics.steps,
                collisions=run.metrics.collisions,
            )
        )
    return rows


def _save_success_figure(summaries: list[dict[str, object]], output_path: Path) -> None:
    labels = {
        "moth": "Moth",
        "bc": "Behavior cloning",
        "airl": "AIRL (BC init)",
        "ppo_normal": "Normal PPO",
        "ppo_robust": "Robust PPO",
    }
    colors = {
        "moth": "#2563a8",
        "bc": "#7c3aed",
        "airl": "#c2410c",
        "ppo_normal": "#0891b2",
        "ppo_robust": "#15803d",
    }
    policies = [str(row["policy"]) for row in summaries]
    values = [float(row["success_rate"]) for row in summaries]
    lower_errors = [
        value - float(row["success_ci95_low"]) for value, row in zip(values, summaries, strict=True)
    ]
    upper_errors = [
        float(row["success_ci95_high"]) - value
        for value, row in zip(values, summaries, strict=True)
    ]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.bar(
        [labels[policy] for policy in policies],
        values,
        color=[colors[policy] for policy in policies],
        yerr=np.asarray((lower_errors, upper_errors)),
        capsize=4,
    )
    axis.set_ylim(0, 105)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Policies learned from synthetic moth demonstrations")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.demonstrations,
        args.validation_episodes,
        args.evaluation_episodes,
        args.airl_rounds,
        args.round_timesteps,
        args.generator_samples,
        args.n_envs,
        args.max_steps,
    )
    if any(value <= 0 for value in positive):
        raise SystemExit("Experiment sizes must be positive.")
    for path in (args.normal_model, args.robust_model):
        if not path.exists():
            raise SystemExit(f"Model not found: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = args.results_dir / "data"
    figure_dir = args.results_dir / "figures"
    args.models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = data_dir / "irl_protocol.json"
    protocol = {
        "question": (
            "Can synthetic moth-controller demonstrations produce a useful policy and "
            "reward without source coordinates or distance?"
        ),
        "method": (
            "class-balanced behavior cloning followed by BC-initialized adversarial IRL (AIRL)"
        ),
        "demonstrator": "synthetic moth-inspired finite-state controller",
        "successful_demonstrations": args.demonstrations,
        "demo_seed_start": args.demo_seed_start,
        "held_out_demo_seed_start": args.held_out_demo_seed_start,
        "validation_seeds": [
            args.validation_seed_start,
            args.validation_seed_start + args.validation_episodes - 1,
        ],
        "evaluation_seeds": [
            args.evaluation_seed_start,
            args.evaluation_seed_start + args.evaluation_episodes - 1,
        ],
        "policy_observation_values": 13,
        "source_position_observed": False,
        "source_distance_observed": False,
        "environment_reward_used_by_bc_or_airl": False,
        "source_contact_terminates_episode": True,
        "physical_success_used_for_model_selection": True,
        "airl_rounds": args.airl_rounds,
        "round_timesteps": args.round_timesteps,
        "generator_samples_per_round": args.generator_samples,
        "airl_initialization": "behavior-cloning policy weights",
        "discriminator_epochs_per_round": 2,
        "discriminator_learning_rate": 0.0001,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "claim_boundary": (
            "This is reward inference from a synthetic controller, not learning from "
            "animals and not evidence of biological reward recovery."
        ),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote protocol to {protocol_path}", flush=True)

    demonstrations, attempts = collect_moth_demonstrations(
        successful_episodes=args.demonstrations,
        seed_start=args.demo_seed_start,
        max_steps=args.max_steps,
    )
    held_out_demonstrations, held_out_attempts = collect_moth_demonstrations(
        successful_episodes=20,
        seed_start=args.held_out_demo_seed_start,
        max_steps=args.max_steps,
    )
    attempt_rows = [{"split": "train", **asdict(row)} for row in attempts] + [
        {"split": "held_out", **asdict(row)} for row in held_out_attempts
    ]
    _write_rows(data_dir / "irl_demonstrations.csv", attempt_rows)
    print(
        f"Collected {args.demonstrations} demonstrations with {len(demonstrations)} transitions",
        flush=True,
    )

    bc_model = make_ppo(seed=args.seed, n_envs=1, max_steps=args.max_steps)
    bc_losses = behavior_clone(
        bc_model,
        demonstrations,
        epochs=30,
        seed=args.seed,
    )
    held_out_accuracy = action_accuracy(bc_model, held_out_demonstrations)
    validation_seeds = range(
        args.validation_seed_start,
        args.validation_seed_start + args.validation_episodes,
    )
    bc_validation = evaluate_policy(
        bc_model,
        policy_name="bc",
        seeds=validation_seeds,
        max_steps=args.max_steps,
    )
    bc_model.save(args.models_dir / "bc_policy")
    print(
        f"BC held-out action accuracy={held_out_accuracy:.3f}; "
        f"validation success={summarize_policy(bc_validation)['success_rate']:.1f}%",
        flush=True,
    )

    generator = make_ppo(
        seed=args.seed + 1,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
    )
    generator.policy.load_state_dict(bc_model.policy.state_dict())
    reward_network = AIRLRewardNetwork(
        observation_size=13,
        action_count=6,
        hidden_size=64,
        gamma=0.995,
    )
    attach_airl_reward_environment(
        generator,
        reward_network,
        n_envs=args.n_envs,
        seed=args.seed + 10_000,
        max_steps=args.max_steps,
    )
    training_rows: list[dict[str, object]] = []
    best_key = (-1.0, float("-inf"))
    best_round = 0
    best_model_path = args.models_dir / "airl_policy"
    best_reward_path = args.models_dir / "airl_reward.pt"
    for round_index in range(1, args.airl_rounds + 1):
        generated = collect_policy_transitions(
            generator,
            minimum_transitions=args.generator_samples,
            seed_start=20_000 + round_index * 1_000,
            max_steps=args.max_steps,
            deterministic=False,
        )
        discriminator = train_airl_discriminator(
            reward_network,
            generator,
            demonstrations,
            generated,
            epochs=2,
            batch_size=512,
            learning_rate=1e-4,
            seed=args.seed + round_index,
        )
        generator.learn(total_timesteps=args.round_timesteps, reset_num_timesteps=False)
        validation = evaluate_policy(
            generator,
            policy_name="airl",
            seeds=validation_seeds,
            max_steps=args.max_steps,
        )
        summary = summarize_policy(validation)
        key = _selection_key(validation)
        if key > best_key:
            best_key = key
            best_round = round_index
            generator.save(best_model_path)
            torch.save(reward_network.state_dict(), best_reward_path)
        row = {
            "round": round_index,
            "generator_timesteps": generator.num_timesteps,
            "generated_transitions": len(generated),
            "discriminator_loss": discriminator.loss,
            "discriminator_accuracy": discriminator.accuracy,
            "validation_success_rate": summary["success_rate"],
            "validation_mean_success_steps": summary["mean_success_steps"],
            "selected": round_index == best_round,
        }
        training_rows.append(row)
        print(
            f"AIRL round {round_index:02d}: disc_acc={discriminator.accuracy:.3f}, "
            f"validation={summary['success_rate']:.1f}%",
            flush=True,
        )

    for row in training_rows:
        row["selected"] = row["round"] == best_round
    _write_rows(data_dir / "irl_training.csv", training_rows)
    generator.get_env().close()
    bc_model.get_env().close()
    selected_airl = PPO.load(best_model_path.with_suffix(".zip"), device="cpu")

    evaluation_seeds = range(
        args.evaluation_seed_start,
        args.evaluation_seed_start + args.evaluation_episodes,
    )
    normal_model = PPO.load(args.normal_model, device="cpu")
    robust_model = PPO.load(args.robust_model, device="cpu")
    all_rows = _moth_rows(evaluation_seeds, args.max_steps)
    all_rows.extend(
        evaluate_policy(
            PPO.load((args.models_dir / "bc_policy.zip"), device="cpu"),
            policy_name="bc",
            seeds=evaluation_seeds,
            max_steps=args.max_steps,
        )
    )
    all_rows.extend(
        evaluate_policy(
            selected_airl,
            policy_name="airl",
            seeds=evaluation_seeds,
            max_steps=args.max_steps,
        )
    )
    all_rows.extend(
        evaluate_policy(
            normal_model,
            policy_name="ppo_normal",
            seeds=evaluation_seeds,
            max_steps=args.max_steps,
        )
    )
    all_rows.extend(
        evaluate_policy(
            robust_model,
            policy_name="ppo_robust",
            seeds=evaluation_seeds,
            max_steps=args.max_steps,
        )
    )
    episode_rows = [asdict(row) for row in all_rows]
    _write_rows(data_dir / "irl_evaluation_episodes.csv", episode_rows)
    policy_order = ("moth", "bc", "airl", "ppo_normal", "ppo_robust")
    summaries = [
        summarize_policy([row for row in all_rows if row.policy == policy])
        for policy in policy_order
    ]
    bc_final_rows = [row for row in all_rows if row.policy == "bc"]
    airl_final_rows = [row for row in all_rows if row.policy == "airl"]
    bc_airl_comparison = compare_paired_success(bc_final_rows, airl_final_rows)
    _write_rows(data_dir / "irl_evaluation_summary.csv", summaries)
    _save_success_figure(summaries, figure_dir / "irl_success_rate.png")
    analysis = {
        "protocol": str(protocol_path),
        "demonstration_attempts": len(attempts),
        "demonstration_successes_retained": args.demonstrations,
        "demonstration_transitions": len(demonstrations),
        "bc_final_training_loss": bc_losses[-1],
        "bc_held_out_action_accuracy": held_out_accuracy,
        "bc_validation": summarize_policy(bc_validation),
        "airl_selected_round": best_round,
        "airl_validation_success_rate": 100.0 * best_key[0],
        "final_evaluation": summaries,
        "bc_airl_paired_success": bc_airl_comparison,
        "interpretation_rule": (
            "AIRL is reported separately from its BC initialization. It is treated as an "
            "improvement only if final success exceeds BC without increasing collisions."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    (data_dir / "irl_analysis.json").write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(analysis, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
