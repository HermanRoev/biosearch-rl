"""Core world dynamics shared by controllers and reinforcement learning.

``BioSearchSimulation`` contains the physics without a Gymnasium dependency.
:mod:`biosearch.gym_environment` adds the RL API around the same dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import pi

import numpy as np
from numpy.typing import NDArray

from biosearch.config import Rectangle, SimulationConfig
from biosearch.plume import PuffPlume
from biosearch.sensors import BilateralOdorSensor, SensorReading

FloatArray = NDArray[np.float64]


class Action(IntEnum):
    """Discrete mobile-agent controls reserved for all project milestones."""

    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    SHARP_LEFT = 3
    SHARP_RIGHT = 4
    STILL = 5


@dataclass(frozen=True)
class StepResult:
    """Values produced by one simulation step."""

    sensor_reading: SensorReading
    collision: bool
    success: bool
    terminated: bool
    truncated: bool
    step_count: int


def _circle_intersects_rectangle(
    center: FloatArray,
    radius: float,
    rectangle: Rectangle,
) -> bool:
    nearest_x = np.clip(center[0], rectangle.x, rectangle.x + rectangle.width)
    nearest_y = np.clip(center[1], rectangle.y, rectangle.y + rectangle.height)
    delta_x = center[0] - nearest_x
    delta_y = center[1] - nearest_y
    return bool(delta_x * delta_x + delta_y * delta_y < radius * radius)


class BioSearchSimulation:
    """Bounded 2D world containing an agent, source, obstacles, and odor plume."""

    def __init__(self, config: SimulationConfig | None = None, seed: int | None = None) -> None:
        self.config = config or SimulationConfig()
        self.seed = seed

        self.position = np.asarray(self.config.world.agent_start, dtype=np.float64)
        self.heading = self.config.world.agent_start_heading
        self.step_count = 0
        self.collision_count = 0
        self.success = False
        self.terminated = False
        self.truncated = False
        self.previous_action = Action.STILL
        self.trajectory: list[FloatArray] = []
        self.action_history: list[Action] = []
        self.detection_history: list[bool] = []
        self.sensor_history: list[tuple[float, float]] = []
        self.collision_history: list[bool] = []
        self.last_sensor_reading: SensorReading

        self._reset_random_generators(seed)
        self.plume = PuffPlume(self.config.plume, self.config.world, self._plume_rng)
        self.sensors = BilateralOdorSensor(
            self.config.sensors,
            self.config.agent,
            self._sensor_rng,
        )
        self.reset(seed=seed)

    def _reset_random_generators(self, seed: int | None) -> None:
        seed_sequence = np.random.SeedSequence(seed)
        plume_seed, sensor_seed = seed_sequence.spawn(2)
        self._plume_rng = np.random.default_rng(plume_seed)
        self._sensor_rng = np.random.default_rng(sensor_seed)

    def reset(
        self,
        seed: int | None = None,
        *,
        agent_position: tuple[float, float] | None = None,
        heading: float | None = None,
    ) -> SensorReading:
        """Reset world state and return the initial sensor reading.

        Passing the same seed and initial pose produces the same future plume
        and sensor sequence.
        """

        if seed is not None or not hasattr(self, "_plume_rng"):
            self.seed = seed
            self._reset_random_generators(seed)

        self.plume = PuffPlume(self.config.plume, self.config.world, self._plume_rng)
        self.sensors = BilateralOdorSensor(
            self.config.sensors,
            self.config.agent,
            self._sensor_rng,
        )
        self.position = np.asarray(
            agent_position or self.config.world.agent_start, dtype=np.float64
        ).copy()
        self.heading = self._wrap_angle(
            self.config.world.agent_start_heading if heading is None else heading
        )
        self._validate_pose(self.position)
        self.step_count = 0
        self.collision_count = 0
        self.success = self._reached_source()
        self.terminated = self.success
        self.truncated = False
        self.previous_action = Action.STILL
        self.trajectory = [self.position.copy()]
        self.last_sensor_reading = self.sensors.read(self.position, self.heading, self.plume)
        self.action_history = []
        self.detection_history = [self.last_sensor_reading.detected]
        self.sensor_history = [(self.last_sensor_reading.left, self.last_sensor_reading.right)]
        self.collision_history = []
        return self.last_sensor_reading

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + pi) % (2 * pi) - pi)

    def _validate_pose(self, position: FloatArray) -> None:
        if position.shape != (2,):
            raise ValueError("agent_position must contain exactly two coordinates.")
        if self._position_collides(position):
            raise ValueError("Initial agent pose collides with a boundary or obstacle.")

    def _position_collides(self, position: FloatArray) -> bool:
        radius = self.config.agent.radius
        world = self.config.world
        outside = (
            position[0] - radius < 0
            or position[0] + radius > world.width
            or position[1] - radius < 0
            or position[1] + radius > world.height
        )
        if outside:
            return True
        return any(
            _circle_intersects_rectangle(position, radius, obstacle) for obstacle in world.obstacles
        )

    def _reached_source(self) -> bool:
        source = np.asarray(self.config.world.source_position, dtype=np.float64)
        distance = float(np.linalg.norm(self.position - source))
        return distance <= self.config.world.source_radius

    def step(self, action: Action | int) -> StepResult:
        """Advance agent, plume, and sensors by one step."""

        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step a finished simulation; call reset().")
        try:
            selected_action = Action(action)
        except ValueError as error:
            raise ValueError(f"Unknown action {action!r}.") from error

        turn = 0.0
        move_forward = False
        if selected_action == Action.FORWARD:
            move_forward = True
        elif selected_action == Action.TURN_LEFT:
            turn = self.config.agent.gentle_turn
            move_forward = True
        elif selected_action == Action.TURN_RIGHT:
            turn = -self.config.agent.gentle_turn
            move_forward = True
        elif selected_action == Action.SHARP_LEFT:
            turn = self.config.agent.sharp_turn
        elif selected_action == Action.SHARP_RIGHT:
            turn = -self.config.agent.sharp_turn

        self.heading = self._wrap_angle(self.heading + turn)
        collision = False
        if move_forward:
            direction = np.array([np.cos(self.heading), np.sin(self.heading)], dtype=np.float64)
            candidate = self.position + self.config.agent.forward_speed * direction
            collision = self._position_collides(candidate)
            if collision:
                self.collision_count += 1
            else:
                self.position = candidate

        self.plume.step()
        self.last_sensor_reading = self.sensors.read(self.position, self.heading, self.plume)
        self.previous_action = selected_action
        self.step_count += 1
        self.trajectory.append(self.position.copy())
        self.action_history.append(selected_action)
        self.detection_history.append(self.last_sensor_reading.detected)
        self.sensor_history.append((self.last_sensor_reading.left, self.last_sensor_reading.right))
        self.collision_history.append(collision)

        self.success = self._reached_source()
        self.terminated = self.success
        self.truncated = self.step_count >= self.config.agent.max_steps and not self.success
        return StepResult(
            sensor_reading=self.last_sensor_reading,
            collision=collision,
            success=self.success,
            terminated=self.terminated,
            truncated=self.truncated,
            step_count=self.step_count,
        )
