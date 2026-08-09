"""Reproducible Milestone 4 controller comparisons."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO

from biosearch.config import SimulationConfig
from biosearch.evaluation import (
    MILESTONE4_CONDITIONS,
    AggregateMetrics,
    ControllerName,
    EpisodeMetrics,
    EpisodeRun,
    SensorCondition,
    calculate_metrics,
    config_for_condition,
    run_episode,
    sample_start_pose,
)
from biosearch.gym_environment import BioSearchEnv

FloatArray = NDArray[np.float64]
ProgressCallback = Callable[[str], None]
ObservationTransform = Callable[[NDArray[np.float32]], NDArray[np.float32]]
ObservationTransformFactory = Callable[[int], ObservationTransform]


@dataclass(frozen=True)
class EpisodeSetup:
    """Per-seed environment configuration and fixed initial pose."""

    config: SimulationConfig
    position: tuple[float, float]
    heading: float


EpisodeSetupFactory = Callable[[int, SimulationConfig], EpisodeSetup]


@dataclass(frozen=True)
class ExperimentResults:
    """Episode rows and controller visitation counts."""

    runs: tuple[EpisodeRun, ...]
    visitation: dict[tuple[str, str], FloatArray]
    x_edges: FloatArray
    y_edges: FloatArray

    @property
    def metrics(self) -> list[EpisodeMetrics]:
        """Return scalar rows in experiment order."""

        return [run.metrics for run in self.runs]


def experiment_config(max_steps: int) -> SimulationConfig:
    """Return a shared episode limit for every evaluated controller."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    base = SimulationConfig()
    return replace(base, agent=replace(base.agent, max_steps=max_steps))


def evaluate_policy_batch(
    model: PPO,
    *,
    controller_name: ControllerName,
    condition: SensorCondition,
    seeds: Sequence[int],
    base_config: SimulationConfig,
    observation_transform: ObservationTransform | None = None,
    observation_transform_factory: ObservationTransformFactory | None = None,
    episode_setup_factory: EpisodeSetupFactory | None = None,
) -> list[EpisodeRun]:
    """Evaluate one policy with batched inference and paired episode seeds."""

    if controller_name not in (ControllerName.PPO_NORMAL, ControllerName.PPO_ROBUST):
        raise ValueError("controller_name must identify one of the two PPO policies.")
    if observation_transform is not None and observation_transform_factory is not None:
        raise ValueError(
            "Provide either observation_transform or observation_transform_factory, not both."
        )
    config = config_for_condition(condition, base_config)
    environments: list[BioSearchEnv] = []
    observations: list[NDArray[np.float32]] = []
    transforms: list[ObservationTransform | None] = []
    poses: list[tuple[tuple[float, float], float]] = []
    active = np.ones(len(seeds), dtype=np.bool_)
    try:
        for seed in seeds:
            transform = (
                observation_transform_factory(seed)
                if observation_transform_factory is not None
                else observation_transform
            )
            if episode_setup_factory is None:
                position, heading = sample_start_pose(seed, config)
                episode_config = config
            else:
                setup = episode_setup_factory(seed, config)
                position, heading = setup.position, setup.heading
                episode_config = setup.config
            environment = BioSearchEnv(episode_config, randomize_start=False)
            observation, _ = environment.reset(
                seed=seed,
                options={
                    "agent_position": position,
                    "heading": heading,
                    "simulation_seed": seed,
                },
            )
            environments.append(environment)
            observations.append(transform(observation) if transform is not None else observation)
            transforms.append(transform)
            poses.append((position, heading))

        while np.any(active):
            active_indices = np.flatnonzero(active)
            observation_batch = np.stack([observations[index] for index in active_indices])
            action_batch, _ = model.predict(observation_batch, deterministic=True)
            for index, action in zip(active_indices, action_batch, strict=True):
                observation, _, terminated, truncated, _ = environments[index].step(
                    int(np.asarray(action).item())
                )
                transform = transforms[index]
                observations[index] = (
                    transform(observation) if transform is not None else observation
                )
                active[index] = not (terminated or truncated)

        runs: list[EpisodeRun] = []
        for seed, environment, (position, heading) in zip(
            seeds,
            environments,
            poses,
            strict=True,
        ):
            simulation = environment.simulation
            runs.append(
                EpisodeRun(
                    metrics=calculate_metrics(
                        simulation,
                        controller=controller_name,
                        condition=condition,
                        seed=seed,
                    ),
                    trajectory=np.asarray(simulation.trajectory, dtype=np.float64),
                    detections=np.asarray(
                        simulation.detection_history,
                        dtype=np.bool_,
                    ),
                    sensor_values=np.asarray(
                        simulation.sensor_history,
                        dtype=np.float64,
                    ),
                    controller_states=(),
                    start_position=position,
                    start_heading=heading,
                )
            )
        return runs
    finally:
        for environment in environments:
            environment.close()


def run_milestone4_experiments(
    normal_model_path: Path,
    robust_model_path: Path,
    *,
    episodes: int = 50,
    seed_start: int = 3_000,
    max_steps: int = 600,
    conditions: Sequence[SensorCondition] = MILESTONE4_CONDITIONS,
    progress: ProgressCallback | None = None,
    heatmap_bins: tuple[int, int] = (50, 30),
) -> ExperimentResults:
    """Run the paired four-controller, six-condition evaluation matrix."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if not normal_model_path.exists() or not robust_model_path.exists():
        raise FileNotFoundError("Both normal and robust PPO checkpoints are required.")
    base_config = experiment_config(max_steps)
    seeds = tuple(range(seed_start, seed_start + episodes))
    models = {
        ControllerName.PPO_NORMAL: PPO.load(normal_model_path, device="cpu"),
        ControllerName.PPO_ROBUST: PPO.load(robust_model_path, device="cpu"),
    }
    all_runs: list[EpisodeRun] = []
    for condition in conditions:
        for controller in (ControllerName.RANDOM, ControllerName.MOTH):
            if progress is not None:
                progress(f"{condition.value}: {controller.value}")
            all_runs.extend(
                run_episode(
                    controller,
                    condition,
                    seed,
                    base_config=base_config,
                )
                for seed in seeds
            )
        for controller, model in models.items():
            if progress is not None:
                progress(f"{condition.value}: {controller.value}")
            all_runs.extend(
                evaluate_policy_batch(
                    model,
                    controller_name=controller,
                    condition=condition,
                    seeds=seeds,
                    base_config=base_config,
                )
            )

    x_edges = np.linspace(0.0, base_config.world.width, heatmap_bins[0] + 1)
    y_edges = np.linspace(0.0, base_config.world.height, heatmap_bins[1] + 1)
    visitation: dict[tuple[str, str], FloatArray] = {}
    for run in all_runs:
        key = (run.metrics.controller, run.metrics.condition)
        counts, _, _ = np.histogram2d(
            run.trajectory[:, 0],
            run.trajectory[:, 1],
            bins=(x_edges, y_edges),
        )
        if key not in visitation:
            visitation[key] = np.zeros_like(counts)
        visitation[key] += counts
    return ExperimentResults(
        runs=tuple(all_runs),
        visitation=visitation,
        x_edges=x_edges,
        y_edges=y_edges,
    )


def save_experiment_results(
    results: ExperimentResults,
    *,
    csv_path: Path,
    visitation_path: Path,
) -> None:
    """Write scalar episode rows and numeric visitation matrices."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = results.metrics
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].as_dict()))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)

    visitation_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, FloatArray] = {
        "x_edges": results.x_edges,
        "y_edges": results.y_edges,
    }
    arrays.update(
        {
            f"{controller}__{condition}": counts
            for (controller, condition), counts in results.visitation.items()
        }
    )
    np.savez_compressed(visitation_path, **arrays)


def save_aggregate_results(
    aggregates: Sequence[AggregateMetrics],
    output_path: Path,
) -> None:
    """Write aggregate metrics, including 95% success-rate intervals."""

    if not aggregates:
        raise ValueError("At least one aggregate row is required.")
    rows = [asdict(row) for row in aggregates]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
