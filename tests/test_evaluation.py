from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from biosearch.config import AgentConfig, SimulationConfig
from biosearch.environment import Action, BioSearchSimulation
from biosearch.evaluation import (
    CHANGED_WIND_ANGLE,
    NEW_OBSTACLE_LAYOUT,
    ControllerName,
    EpisodeMetrics,
    SensorCondition,
    aggregate_metrics,
    calculate_metrics,
    config_for_condition,
    run_episode,
    sample_domain_randomized_config,
    sample_start_pose,
)


def test_sensor_condition_disables_only_requested_sensor() -> None:
    normal = config_for_condition(SensorCondition.NORMAL)
    failed = config_for_condition(SensorCondition.LEFT_DISABLED)

    assert normal.sensors.disabled_sensor is None
    assert failed.sensors.disabled_sensor == "left"
    assert failed.world == normal.world


def test_milestone4_conditions_change_only_the_declared_domain_features() -> None:
    normal = config_for_condition(SensorCondition.NORMAL)
    noisy = config_for_condition(SensorCondition.MODERATE_NOISE)
    dropout = config_for_condition(SensorCondition.SENSOR_DROPOUT)
    changed_wind = config_for_condition(SensorCondition.CHANGED_WIND)
    new_obstacles = config_for_condition(SensorCondition.NEW_OBSTACLES)

    assert noisy.sensors.noise_std == 0.10
    assert noisy.sensors.dropout_probability == normal.sensors.dropout_probability
    assert dropout.sensors.dropout_probability == 0.25
    assert dropout.sensors.noise_std == normal.sensors.noise_std
    assert changed_wind.world.wind_direction == pytest.approx(CHANGED_WIND_ANGLE)
    assert changed_wind.world.agent_start != normal.world.agent_start
    assert new_obstacles.world.obstacles == NEW_OBSTACLE_LAYOUT


def test_domain_randomization_is_seeded_and_within_documented_ranges() -> None:
    base = SimulationConfig()
    first = sample_domain_randomized_config(np.random.default_rng(23), base)
    second = sample_domain_randomized_config(np.random.default_rng(23), base)

    assert first == second
    for seed in range(50):
        sampled = sample_domain_randomized_config(np.random.default_rng(seed), base)
        assert 0.01 <= sampled.sensors.noise_std <= 0.10
        assert 0.0 <= sampled.sensors.dropout_probability <= 0.25
        assert -CHANGED_WIND_ANGLE <= sampled.world.wind_direction <= CHANGED_WIND_ANGLE
        assert sampled.world.obstacles in (base.world.obstacles, NEW_OBSTACLE_LAYOUT)


def test_start_pose_is_seeded_and_inside_world() -> None:
    config = SimulationConfig()

    first = sample_start_pose(17, config)
    second = sample_start_pose(17, config)

    assert first == second
    (x, y), _ = first
    assert 0 < x < config.world.width
    assert 0 < y < config.world.height


def test_metrics_use_recorded_trajectory() -> None:
    config = SimulationConfig(agent=replace(AgentConfig(), max_steps=3))
    simulation = BioSearchSimulation(config, seed=5)
    simulation.step(Action.FORWARD)
    simulation.step(Action.SHARP_LEFT)
    simulation.step(Action.STILL)

    metrics = calculate_metrics(
        simulation,
        controller=ControllerName.MOTH,
        condition=SensorCondition.NORMAL,
        seed=5,
    )

    assert metrics.steps == 3
    assert metrics.path_length == pytest.approx(config.agent.forward_speed)
    assert metrics.tortuosity == pytest.approx(1.0)
    assert metrics.collisions == 0


def test_run_episode_respects_short_time_limit() -> None:
    base = SimulationConfig(agent=replace(AgentConfig(), max_steps=8))

    result = run_episode(
        ControllerName.MOTH,
        SensorCondition.NORMAL,
        seed=3,
        base_config=base,
    )

    assert result.metrics.steps == 8
    assert not result.metrics.success
    assert result.trajectory.shape == (9, 2)
    assert result.detections.shape == (9,)
    assert len(result.controller_states) == 8


def test_aggregate_metrics_groups_controller_and_condition() -> None:
    rows = [
        EpisodeMetrics("moth", "normal", 1, True, 10, 2.0, 0, 1.2, 30.0, 0.2),
        EpisodeMetrics("moth", "normal", 2, False, 20, 3.0, 2, 1.8, 10.0, 4.0),
    ]

    aggregate = aggregate_metrics(rows)

    assert len(aggregate) == 1
    assert aggregate[0].success_rate == 50.0
    assert aggregate[0].success_ci95_low < 50.0 < aggregate[0].success_ci95_high
    assert aggregate[0].mean_success_steps == 10.0
    assert aggregate[0].mean_path_length == 2.5
    assert aggregate[0].mean_collisions == 1.0
    assert np.isfinite(aggregate[0].mean_tortuosity)
