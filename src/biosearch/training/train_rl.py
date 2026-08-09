"""Train a PPO policy with checkpoints and separate seeded evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from biosearch.config import SimulationConfig
from biosearch.evaluation import sample_domain_randomized_config
from biosearch.gym_environment import BioSearchEnv, RewardConfig


def training_config(max_steps: int) -> SimulationConfig:
    """Return the simulation configuration used for the first policy."""

    base = SimulationConfig()
    return replace(base, agent=replace(base.agent, max_steps=max_steps))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BioSearch PPO policy.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/ppo"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Randomize sensors, wind, and obstacles independently each episode.",
    )
    return parser


def train(
    *,
    total_timesteps: int,
    n_envs: int,
    seed: int,
    max_steps: int,
    checkpoint_freq: int,
    eval_freq: int,
    eval_episodes: int,
    models_dir: Path,
    logs_dir: Path,
    device: str,
    domain_randomization: bool = False,
) -> Path:
    """Train PPO, returning the final model path."""

    if total_timesteps <= 0 or n_envs <= 0 or max_steps <= 0:
        raise ValueError("Training sizes must be positive.")
    if checkpoint_freq <= 0 or eval_freq <= 0 or eval_episodes <= 0:
        raise ValueError("Callback frequencies and evaluation episodes must be positive.")

    config = training_config(max_steps)
    reward_config = RewardConfig()
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = models_dir / "checkpoints"
    best_dir = models_dir / "best"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = logs_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    def make_environment() -> BioSearchEnv:
        return BioSearchEnv(
            config,
            reward_config=reward_config,
            randomize_start=True,
            config_sampler=(sample_domain_randomized_config if domain_randomization else None),
        )

    train_env = make_vec_env(
        make_environment,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir),
    )
    eval_env = make_vec_env(
        make_environment,
        n_envs=1,
        seed=seed + 10_000,
    )
    callbacks = [
        CheckpointCallback(
            save_freq=max(checkpoint_freq // n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_biosearch",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(logs_dir / "evaluations"),
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        ),
    ]
    policy_kwargs = {"net_arch": [128, 128]}
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device=device,
    )
    model.set_logger(configure(str(logs_dir), ["stdout", "csv"]))

    started = perf_counter()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        final_model = models_dir / "ppo_biosearch_final"
        model.save(final_model)
    finally:
        train_env.close()
        eval_env.close()
    elapsed = perf_counter() - started

    metadata = {
        "algorithm": "PPO",
        "total_timesteps_requested": total_timesteps,
        "total_timesteps_actual": model.num_timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "max_episode_steps": max_steps,
        "domain_randomization": domain_randomization,
        "domain_randomization_ranges": (
            {
                "sensor_noise_std": [0.01, 0.10],
                "sensor_dropout_probability": [0.0, 0.25],
                "disabled_sensor_sampling": {
                    "both_working": 0.60,
                    "left_disabled": 0.20,
                    "right_disabled": 0.20,
                },
                "wind_direction_radians": [-float(np.pi / 12), float(np.pi / 12)],
                "obstacle_layouts": ["default", "milestone4_new"],
            }
            if domain_randomization
            else None
        ),
        "elapsed_seconds": elapsed,
        "simulation_config": asdict(config),
        "reward_config": asdict(reward_config),
        "ppo_hyperparameters": {
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "net_arch": [128, 128],
        },
        "versions": {
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
    }
    evaluations_path = logs_dir / "evaluations" / "evaluations.npz"
    if evaluations_path.exists():
        with np.load(evaluations_path) as evaluations:
            mean_rewards = evaluations["results"].mean(axis=1)
            best_index = int(np.argmax(mean_rewards))
            metadata["best_periodic_evaluation"] = {
                "timestep": int(evaluations["timesteps"][best_index]),
                "mean_reward": float(mean_rewards[best_index]),
                "success_rate": float(evaluations["successes"][best_index].mean()),
                "episodes": int(evaluations["results"].shape[1]),
            }
    (models_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return final_model.with_suffix(".zip")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        max_steps=args.max_steps,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir,
        device=args.device,
        domain_randomization=args.domain_randomization,
    )
    print(f"Saved final model to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
