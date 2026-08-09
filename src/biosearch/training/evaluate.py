"""Evaluate and animate a saved PPO policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from biosearch.config import SimulationConfig
from biosearch.evaluation import (
    ControllerName,
    EpisodeMetrics,
    SensorCondition,
    calculate_metrics,
    config_for_condition,
    sample_start_pose,
)
from biosearch.gym_environment import BioSearchEnv
from biosearch.visualization.renderer import PygameRenderer


@dataclass(frozen=True)
class PolicyEvaluation:
    """One deterministic policy episode."""

    metrics: EpisodeMetrics
    total_reward: float


def policy_config(max_steps: int) -> SimulationConfig:
    """Return the evaluation configuration matching training episode length."""

    base = SimulationConfig()
    return replace(base, agent=replace(base.agent, max_steps=max_steps))


def evaluate_policy_episodes(
    model_path: Path,
    *,
    episodes: int,
    seed_start: int,
    condition: SensorCondition = SensorCondition.NORMAL,
    controller_name: ControllerName = ControllerName.RL,
    max_steps: int = 600,
    varied_start: bool = True,
) -> list[PolicyEvaluation]:
    """Run deterministic policy episodes without renderer overhead."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    model = PPO.load(model_path, device="cpu")
    config = config_for_condition(condition, policy_config(max_steps))
    environment = BioSearchEnv(config, randomize_start=False)
    evaluations: list[PolicyEvaluation] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            position, heading = (
                sample_start_pose(seed, config)
                if varied_start
                else (config.world.agent_start, config.world.agent_start_heading)
            )
            observation, _ = environment.reset(
                seed=seed,
                options={
                    "agent_position": position,
                    "heading": heading,
                    "simulation_seed": seed,
                },
            )
            total_reward = 0.0
            terminated = truncated = False
            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = environment.step(
                    int(np.asarray(action).item())
                )
                total_reward += reward
            metrics = calculate_metrics(
                environment.simulation,
                controller=controller_name,
                condition=condition,
                seed=seed,
            )
            evaluations.append(PolicyEvaluation(metrics=metrics, total_reward=total_reward))
    finally:
        environment.close()
    return evaluations


def save_policy_gif(
    model_path: Path,
    *,
    seed: int,
    output_path: Path,
    condition: SensorCondition = SensorCondition.NORMAL,
    controller_name: ControllerName = ControllerName.RL,
    max_steps: int = 600,
    varied_start: bool = True,
    frame_stride: int = 3,
    fps: int = 15,
    frame_size: tuple[int, int] = (826, 490),
) -> EpisodeMetrics:
    """Render one deterministic learned-policy episode to a GIF."""

    if frame_stride <= 0 or fps <= 0:
        raise ValueError("frame_stride and fps must be positive.")
    model = PPO.load(model_path, device="cpu")
    config = config_for_condition(condition, policy_config(max_steps))
    environment = BioSearchEnv(config, randomize_start=False)
    position, heading = (
        sample_start_pose(seed, config)
        if varied_start
        else (config.world.agent_start, config.world.agent_start_heading)
    )
    observation, _ = environment.reset(
        seed=seed,
        options={
            "agent_position": position,
            "heading": heading,
            "simulation_seed": seed,
        },
    )
    renderer = PygameRenderer(
        environment.simulation,
        caption=f"{controller_name.value}: {condition.value}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    terminated = truncated = False
    try:
        with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
            renderer.render(controller_state=controller_name.value.upper())
            writer.append_data(renderer.capture_frame(size=frame_size))
            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = environment.step(
                    int(np.asarray(action).item())
                )
                final_step = terminated or truncated
                if environment.simulation.step_count % frame_stride == 0 or final_step:
                    renderer.render(controller_state=controller_name.value.upper())
                    writer.append_data(renderer.capture_frame(size=frame_size))
    finally:
        renderer.close()
        environment.close()
    return calculate_metrics(
        environment.simulation,
        controller=controller_name,
        condition=condition,
        seed=seed,
    )


def summarize_policy_evaluations(
    evaluations: list[PolicyEvaluation],
) -> dict[str, float]:
    """Calculate concise learned-policy summary values."""

    successful = [
        evaluation.metrics.steps for evaluation in evaluations if evaluation.metrics.success
    ]
    return {
        "success_rate": 100.0 * fmean(evaluation.metrics.success for evaluation in evaluations),
        "mean_success_steps": fmean(successful) if successful else float("nan"),
        "mean_reward": fmean(evaluation.total_reward for evaluation in evaluations),
    }
