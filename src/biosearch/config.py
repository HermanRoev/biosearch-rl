"""Typed configuration for the simulation.

Distances are expressed in abstract world units and angles in radians. One
simulation step is also an abstract time unit; the model is deliberately not
calibrated to a particular animal, robot, or physical plume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangular obstacle in world coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Obstacle width and height must be positive.")


@dataclass(frozen=True)
class WorldConfig:
    """Dimensions, source location, wind, and static geometry."""

    width: float = 20.0
    height: float = 12.0
    source_position: tuple[float, float] = (2.5, 6.0)
    source_radius: float = 0.45
    agent_start: tuple[float, float] = (16.5, 6.0)
    agent_start_heading: float = pi
    wind_direction: float = 0.0
    wind_speed: float = 0.10
    obstacles: tuple[Rectangle, ...] = (
        Rectangle(8.0, 2.0, 0.65, 3.0),
        Rectangle(11.5, 7.0, 0.65, 3.0),
    )

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("World width and height must be positive.")
        if self.source_radius <= 0:
            raise ValueError("Source radius must be positive.")
        for label, point in (
            ("source_position", self.source_position),
            ("agent_start", self.agent_start),
        ):
            if not (0 <= point[0] <= self.width and 0 <= point[1] <= self.height):
                raise ValueError(f"{label} must lie inside the world.")


@dataclass(frozen=True)
class AgentConfig:
    """Mobile-agent geometry and kinematics."""

    radius: float = 0.28
    forward_speed: float = 0.16
    gentle_turn: float = pi / 12
    sharp_turn: float = pi / 5
    sensor_forward_offset: float = 0.38
    sensor_lateral_offset: float = 0.24
    max_steps: int = 1_200

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.forward_speed <= 0:
            raise ValueError("Agent radius and forward speed must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")


@dataclass(frozen=True)
class PlumeConfig:
    """Parameters of the inexpensive abstract puff plume."""

    emission_interval: int = 3
    puffs_per_emission: int = 3
    initial_intensity: float = 1.0
    intensity_decay: float = 0.012
    lateral_diffusion: float = 0.055
    longitudinal_jitter: float = 0.018
    influence_radius: float = 0.60
    max_age: int = 240
    max_puffs: int = 600

    def __post_init__(self) -> None:
        if self.emission_interval <= 0 or self.puffs_per_emission <= 0:
            raise ValueError("Puff emission parameters must be positive.")
        if self.initial_intensity <= 0 or self.influence_radius <= 0:
            raise ValueError("Puff intensity and influence radius must be positive.")
        if not 0 <= self.intensity_decay < 1:
            raise ValueError("intensity_decay must be in [0, 1).")
        if self.max_age <= 0 or self.max_puffs <= 0:
            raise ValueError("Puff lifetime and capacity must be positive.")


@dataclass(frozen=True)
class SensorConfig:
    """Noise, dropout, detection, and sensor-failure settings."""

    noise_std: float = 0.015
    dropout_probability: float = 0.02
    detection_threshold: float = 0.08
    disabled_sensor: str | None = None
    max_reading: float = 1.0

    def __post_init__(self) -> None:
        if self.noise_std < 0:
            raise ValueError("noise_std cannot be negative.")
        if not 0 <= self.dropout_probability <= 1:
            raise ValueError("dropout_probability must be in [0, 1].")
        if self.disabled_sensor not in (None, "left", "right"):
            raise ValueError("disabled_sensor must be None, 'left', or 'right'.")
        if self.detection_threshold < 0 or self.max_reading <= 0:
            raise ValueError("Sensor thresholds must be non-negative and finite.")


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level immutable simulation configuration."""

    world: WorldConfig = field(default_factory=WorldConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    plume: PlumeConfig = field(default_factory=PlumeConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
