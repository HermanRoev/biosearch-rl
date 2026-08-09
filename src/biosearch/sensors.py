"""Bilateral virtual odor sensors with configurable uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biosearch.config import AgentConfig, SensorConfig
from biosearch.plume import PuffPlume

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SensorReading:
    """Sensor locations, noisy readings, and a thresholded detection flag."""

    left: float
    right: float
    detected: bool
    left_position: FloatArray
    right_position: FloatArray


def sensor_positions(
    agent_position: FloatArray,
    heading: float,
    agent_config: AgentConfig,
) -> tuple[FloatArray, FloatArray]:
    """Calculate antenna-like sensor locations in world coordinates."""

    position = np.asarray(agent_position, dtype=np.float64)
    forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    left_axis = np.array([-forward[1], forward[0]], dtype=np.float64)
    base = position + agent_config.sensor_forward_offset * forward
    lateral = agent_config.sensor_lateral_offset * left_axis
    return base + lateral, base - lateral


class BilateralOdorSensor:
    """Read the plume at two points and apply failure/noise effects."""

    def __init__(
        self,
        config: SensorConfig,
        agent_config: AgentConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.agent_config = agent_config
        self.rng = rng

    def read(
        self,
        agent_position: FloatArray,
        heading: float,
        plume: PuffPlume,
    ) -> SensorReading:
        """Sample both sensors once."""

        left_position, right_position = sensor_positions(agent_position, heading, self.agent_config)
        values = plume.concentration_at(np.stack((left_position, right_position)))
        values = values + self.rng.normal(0.0, self.config.noise_std, size=2)
        values = np.clip(values, 0.0, self.config.max_reading)

        if self.config.dropout_probability:
            dropped = self.rng.random(2) < self.config.dropout_probability
            values[dropped] = 0.0
        if self.config.disabled_sensor == "left":
            values[0] = 0.0
        elif self.config.disabled_sensor == "right":
            values[1] = 0.0

        left, right = float(values[0]), float(values[1])
        return SensorReading(
            left=left,
            right=right,
            detected=max(left, right) >= self.config.detection_threshold,
            left_position=left_position,
            right_position=right_position,
        )
