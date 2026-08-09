from __future__ import annotations

from dataclasses import replace

import numpy as np

from biosearch.config import AgentConfig, PlumeConfig, SensorConfig, WorldConfig
from biosearch.plume import PuffPlume
from biosearch.sensors import BilateralOdorSensor, sensor_positions


def make_plume(seed: int = 1) -> PuffPlume:
    world = WorldConfig(obstacles=())
    plume = PuffPlume(
        PlumeConfig(emission_interval=1, puffs_per_emission=1),
        world,
        np.random.default_rng(seed),
    )
    plume.step()
    return plume


def test_sensor_positions_rotate_with_agent() -> None:
    agent = AgentConfig(sensor_forward_offset=1.0, sensor_lateral_offset=0.5)

    left, right = sensor_positions(np.array([3.0, 4.0]), 0.0, agent)

    np.testing.assert_allclose(left, [4.0, 4.5])
    np.testing.assert_allclose(right, [4.0, 3.5])


def test_disabled_left_sensor_is_exactly_zero() -> None:
    plume = make_plume()
    config = SensorConfig(
        noise_std=0.0,
        dropout_probability=0.0,
        disabled_sensor="left",
    )
    sensor = BilateralOdorSensor(config, AgentConfig(), np.random.default_rng(2))

    reading = sensor.read(np.array([2.5, 6.0]), 0.0, plume)

    assert reading.left == 0.0
    assert reading.right > 0.0


def test_full_dropout_zeros_both_sensors() -> None:
    plume = make_plume()
    config = SensorConfig(noise_std=0.0, dropout_probability=1.0)
    sensor = BilateralOdorSensor(config, AgentConfig(), np.random.default_rng(2))

    reading = sensor.read(np.array([2.5, 6.0]), 0.0, plume)

    assert reading.left == 0.0
    assert reading.right == 0.0
    assert not reading.detected


def test_seeded_sensor_noise_is_reproducible() -> None:
    plume = make_plume()
    config = replace(SensorConfig(), noise_std=0.2, dropout_probability=0.3)
    first = BilateralOdorSensor(config, AgentConfig(), np.random.default_rng(42))
    second = BilateralOdorSensor(config, AgentConfig(), np.random.default_rng(42))

    first_reading = first.read(np.array([6.0, 6.0]), 0.0, plume)
    second_reading = second.read(np.array([6.0, 6.0]), 0.0, plume)

    assert first_reading.left == second_reading.left
    assert first_reading.right == second_reading.right
