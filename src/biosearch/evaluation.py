"""Seeded episode evaluation and quantitative search metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import pi
from statistics import fmean
from typing import Any

import numpy as np
from numpy.typing import NDArray

from biosearch.config import Rectangle, SimulationConfig, WorldConfig
from biosearch.controllers import MothController, RandomController
from biosearch.environment import Action, BioSearchSimulation

FloatArray = NDArray[np.float64]


class ControllerName(StrEnum):
    """Controllers available to the evaluation runner."""

    RANDOM = "random"
    MOTH = "moth"
    RL = "rl"
    PPO_NORMAL = "ppo_normal"
    PPO_ROBUST = "ppo_robust"


class SensorCondition(StrEnum):
    """Named evaluation scenarios.

    The historical class name is retained for backward compatibility even
    though Milestone 4 also varies wind and obstacle geometry.
    """

    NORMAL = "normal"
    MODERATE_NOISE = "moderate_noise"
    SENSOR_DROPOUT = "sensor_dropout"
    LEFT_DISABLED = "left_disabled"
    RIGHT_DISABLED = "right_disabled"
    CHANGED_WIND = "changed_wind"
    NEW_OBSTACLES = "new_obstacles"


MILESTONE4_CONDITIONS = (
    SensorCondition.NORMAL,
    SensorCondition.MODERATE_NOISE,
    SensorCondition.SENSOR_DROPOUT,
    SensorCondition.LEFT_DISABLED,
    SensorCondition.CHANGED_WIND,
    SensorCondition.NEW_OBSTACLES,
)

NEW_OBSTACLE_LAYOUT = (
    Rectangle(6.0, 5.0, 0.70, 3.2),
    Rectangle(10.0, 1.0, 0.70, 4.0),
    Rectangle(13.0, 7.0, 0.70, 3.2),
)
CHANGED_WIND_ANGLE = pi / 12


Controller = RandomController | MothController


@dataclass(frozen=True)
class EpisodeMetrics:
    """Scalar results for one search episode."""

    controller: str
    condition: str
    seed: int
    success: bool
    steps: int
    path_length: float
    collisions: int
    tortuosity: float
    odor_detection_percentage: float
    final_distance_to_source: float

    def as_dict(self) -> dict[str, Any]:
        """Return a CSV-friendly field mapping."""

        return asdict(self)


@dataclass(frozen=True)
class EpisodeRun:
    """Metrics plus time-series data used by plots and animations."""

    metrics: EpisodeMetrics
    trajectory: FloatArray
    detections: NDArray[np.bool_]
    sensor_values: FloatArray
    controller_states: tuple[str, ...]
    start_position: tuple[float, float]
    start_heading: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Mean results for one controller/condition group."""

    controller: str
    condition: str
    episodes: int
    success_rate: float
    success_ci95_low: float
    success_ci95_high: float
    mean_success_steps: float
    mean_path_length: float
    mean_collisions: float
    mean_tortuosity: float
    mean_odor_detection_percentage: float


def config_for_condition(
    condition: SensorCondition | str,
    base_config: SimulationConfig | None = None,
) -> SimulationConfig:
    """Create the fixed simulation config for one evaluation condition."""

    selected = SensorCondition(condition)
    base = base_config or SimulationConfig()
    sensors = replace(
        base.sensors,
        noise_std=0.015,
        dropout_probability=0.02,
        disabled_sensor=None,
    )
    world = base.world
    if selected is SensorCondition.MODERATE_NOISE:
        sensors = replace(sensors, noise_std=0.10)
    elif selected is SensorCondition.SENSOR_DROPOUT:
        sensors = replace(sensors, dropout_probability=0.25)
    elif selected is SensorCondition.LEFT_DISABLED:
        sensors = replace(sensors, disabled_sensor="left")
    elif selected is SensorCondition.RIGHT_DISABLED:
        sensors = replace(sensors, disabled_sensor="right")
    elif selected is SensorCondition.CHANGED_WIND:
        world = world_for_wind(base.world, CHANGED_WIND_ANGLE)
    elif selected is SensorCondition.NEW_OBSTACLES:
        world = replace(world, obstacles=NEW_OBSTACLE_LAYOUT)
    return replace(base, world=world, sensors=sensors)


def world_for_wind(world: WorldConfig, wind_direction: float) -> WorldConfig:
    """Rotate wind and keep the nominal start the same distance downwind."""

    source = np.asarray(world.source_position, dtype=np.float64)
    original_start = np.asarray(world.agent_start, dtype=np.float64)
    downwind_distance = float(np.linalg.norm(original_start - source))
    direction = np.array(
        [np.cos(wind_direction), np.sin(wind_direction)],
        dtype=np.float64,
    )
    start = source + downwind_distance * direction
    return replace(
        world,
        wind_direction=float(wind_direction),
        agent_start=(float(start[0]), float(start[1])),
        agent_start_heading=float(wind_direction + pi),
    )


def sample_domain_randomized_config(
    rng: np.random.Generator,
    base_config: SimulationConfig,
) -> SimulationConfig:
    """Sample one training domain from documented robustness ranges.

    Randomization exposes the policy to sensor noise, dropout, occasional
    unilateral sensor loss, modest wind rotation, and two obstacle layouts.
    The held-out evaluation conditions remain fixed and reproducible.
    """

    wind_direction = float(rng.uniform(-CHANGED_WIND_ANGLE, CHANGED_WIND_ANGLE))
    disabled_sensor = rng.choice(np.asarray([None, None, None, "left", "right"], dtype=object))
    sensors = replace(
        base_config.sensors,
        noise_std=float(rng.uniform(0.01, 0.10)),
        dropout_probability=float(rng.uniform(0.0, 0.25)),
        disabled_sensor=disabled_sensor,
    )
    world = world_for_wind(base_config.world, wind_direction)
    if bool(rng.integers(0, 2)):
        world = replace(world, obstacles=NEW_OBSTACLE_LAYOUT)
    return replace(base_config, world=world, sensors=sensors)


def sample_start_pose(
    seed: int,
    config: SimulationConfig,
) -> tuple[tuple[float, float], float]:
    """Return a deterministic, mildly varied downwind start pose.

    Starts remain near the expected plume corridor so Milestone 2 evaluates the
    local search behavior rather than a much harder global exploration problem.
    Both controllers receive the same pose for a given seed.
    """

    rng = np.random.default_rng(np.random.SeedSequence([seed, 2_026]))
    nominal_start = np.asarray(config.world.agent_start, dtype=np.float64)
    crosswind_offset = float(rng.uniform(-2.0, 2.0))
    crosswind = np.array(
        [-np.sin(config.world.wind_direction), np.cos(config.world.wind_direction)],
        dtype=np.float64,
    )
    position = nominal_start + crosswind_offset * crosswind
    margin = config.agent.radius + 0.1
    position[0] = np.clip(position[0], margin, config.world.width - margin)
    position[1] = np.clip(position[1], margin, config.world.height - margin)
    heading = float(config.world.agent_start_heading + rng.uniform(-pi / 4, pi / 4))
    return (float(position[0]), float(position[1])), heading


def make_controller(
    name: ControllerName | str,
    *,
    seed: int,
) -> Controller:
    """Construct a controller with a reproducible initial state."""

    selected = ControllerName(name)
    if selected is ControllerName.RANDOM:
        return RandomController(seed=seed)
    if selected is ControllerName.MOTH:
        return MothController()
    raise ValueError("The learned policy must be loaded from a model checkpoint.")


def action_for_controller(
    controller: Controller,
    simulation: BioSearchSimulation,
) -> Action:
    """Request an action using only inputs available to that controller."""

    reading = simulation.last_sensor_reading
    if isinstance(controller, MothController):
        relative_heading = float(
            (simulation.heading - simulation.config.world.wind_direction + pi) % (2 * pi) - pi
        )
        return controller.act(
            reading,
            heading_relative_to_wind=relative_heading,
        )
    return controller.act(reading)


def run_episode(
    controller_name: ControllerName | str,
    condition: SensorCondition | str,
    seed: int,
    *,
    base_config: SimulationConfig | None = None,
    varied_start: bool = True,
) -> EpisodeRun:
    """Run one complete deterministic episode."""

    selected_controller = ControllerName(controller_name)
    selected_condition = SensorCondition(condition)
    config = config_for_condition(selected_condition, base_config)
    simulation = BioSearchSimulation(config, seed=seed)
    if varied_start:
        start_position, start_heading = sample_start_pose(seed, config)
        simulation.reset(
            seed=seed,
            agent_position=start_position,
            heading=start_heading,
        )
    else:
        start_position = config.world.agent_start
        start_heading = config.world.agent_start_heading

    controller = make_controller(selected_controller, seed=seed)
    states: list[str] = []
    while not simulation.terminated and not simulation.truncated:
        action = action_for_controller(controller, simulation)
        states.append(controller.state_name)
        simulation.step(action)

    return EpisodeRun(
        metrics=calculate_metrics(
            simulation,
            controller=selected_controller,
            condition=selected_condition,
            seed=seed,
        ),
        trajectory=np.asarray(simulation.trajectory, dtype=np.float64),
        detections=np.asarray(simulation.detection_history, dtype=np.bool_),
        sensor_values=np.asarray(simulation.sensor_history, dtype=np.float64),
        controller_states=tuple(states),
        start_position=start_position,
        start_heading=start_heading,
    )


def calculate_metrics(
    simulation: BioSearchSimulation,
    *,
    controller: ControllerName | str,
    condition: SensorCondition | str,
    seed: int,
) -> EpisodeMetrics:
    """Calculate path and sensing metrics from recorded simulation history."""

    trajectory = np.asarray(simulation.trajectory, dtype=np.float64)
    segments = np.diff(trajectory, axis=0)
    path_length = float(np.linalg.norm(segments, axis=1).sum())
    displacement = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    tortuosity = path_length / displacement if displacement > 1e-12 else float("nan")
    step_detections = simulation.detection_history[1:]
    detection_percentage = 100.0 * float(np.mean(step_detections)) if step_detections else 0.0
    source = np.asarray(simulation.config.world.source_position, dtype=np.float64)
    final_distance = float(np.linalg.norm(simulation.position - source))
    return EpisodeMetrics(
        controller=ControllerName(controller).value,
        condition=SensorCondition(condition).value,
        seed=seed,
        success=simulation.success,
        steps=simulation.step_count,
        path_length=path_length,
        collisions=simulation.collision_count,
        tortuosity=tortuosity,
        odor_detection_percentage=detection_percentage,
        final_distance_to_source=final_distance,
    )


def aggregate_metrics(metrics: list[EpisodeMetrics]) -> list[AggregateMetrics]:
    """Aggregate episode metrics by controller and sensor condition."""

    grouped: dict[tuple[str, str], list[EpisodeMetrics]] = defaultdict(list)
    for episode in metrics:
        grouped[(episode.controller, episode.condition)].append(episode)

    aggregates: list[AggregateMetrics] = []
    for (controller, condition), episodes in sorted(grouped.items()):
        successes = sum(episode.success for episode in episodes)
        success_rate = 100.0 * successes / len(episodes)
        ci_low, ci_high = success_rate_interval(successes, len(episodes))
        successful_steps = [episode.steps for episode in episodes if episode.success]
        finite_tortuosities = [
            episode.tortuosity for episode in episodes if np.isfinite(episode.tortuosity)
        ]
        aggregates.append(
            AggregateMetrics(
                controller=controller,
                condition=condition,
                episodes=len(episodes),
                success_rate=success_rate,
                success_ci95_low=ci_low,
                success_ci95_high=ci_high,
                mean_success_steps=(fmean(successful_steps) if successful_steps else float("nan")),
                mean_path_length=fmean(episode.path_length for episode in episodes),
                mean_collisions=fmean(episode.collisions for episode in episodes),
                mean_tortuosity=(
                    fmean(finite_tortuosities) if finite_tortuosities else float("nan")
                ),
                mean_odor_detection_percentage=fmean(
                    episode.odor_detection_percentage for episode in episodes
                ),
            )
        )
    return aggregates


def success_rate_interval(successes: int, episodes: int) -> tuple[float, float]:
    """Return a 95% Wilson score interval as percentages."""

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1.0 + z**2 / episodes
    center = (proportion + z**2 / (2 * episodes)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / episodes + z**2 / (4 * episodes**2))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, 100.0 * (center - half_width))
    high = 100.0 if successes == episodes else min(100.0, 100.0 * (center + half_width))
    return low, high
