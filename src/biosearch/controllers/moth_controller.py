"""Readable moth-inspired finite-state search controller.

The controller borrows the broad ideas of surge, zigzag/casting, and looping
from odor-guided insect navigation research. It is a small original engineering
baseline, not a reproduction of a biological model or a
published controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import pi

from biosearch.environment import Action
from biosearch.sensors import SensorReading


class MothState(StrEnum):
    """Behavioral states exposed to logs and the renderer."""

    SURGE = "SURGE"
    ZIGZAG = "ZIGZAG"
    LOOP = "LOOP"


@dataclass(frozen=True)
class MothControllerConfig:
    """Timing and steering parameters for the finite-state controller."""

    zigzag_duration: int = 48
    zigzag_leg_steps: int = 5
    loop_forward_steps: int = 5
    sensor_deadband: float = 0.025
    upwind_tolerance: float = pi / 10

    def __post_init__(self) -> None:
        if self.zigzag_duration <= 0:
            raise ValueError("zigzag_duration must be positive.")
        if self.zigzag_leg_steps <= 0 or self.loop_forward_steps <= 0:
            raise ValueError("Search-pattern timing must be positive.")
        if self.sensor_deadband < 0 or self.upwind_tolerance < 0:
            raise ValueError("Steering thresholds cannot be negative.")


def _wrap_angle(angle: float) -> float:
    return float((angle + pi) % (2 * pi) - pi)


class MothController:
    """Switch among odor-guided surge, local zigzag, and broad looping search.

    State transitions are intentionally explicit:

    * Any odor detection enters ``SURGE``.
    * The first missed reading after a detection enters ``ZIGZAG``.
    * Continued odor absence beyond ``zigzag_duration`` enters ``LOOP``.
    * A new detection from either search state immediately re-enters ``SURGE``.

    ``heading_relative_to_wind`` is local orientation information: zero means
    downwind and ±π means upwind. Source position is never used.
    """

    def __init__(self, config: MothControllerConfig | None = None) -> None:
        self.config = config or MothControllerConfig()
        self.reset()

    @property
    def state_name(self) -> str:
        """Current state label for rendering and logs."""

        return self.state.value

    def reset(self, seed: int | None = None) -> None:
        """Reset state; ``seed`` is accepted for a common controller API."""

        del seed
        # No detection history exists at reset, so begin with the broad search state.
        self.state = MothState.LOOP
        self.steps_since_detection = self.config.zigzag_duration + 1
        self._zigzag_direction = 1
        self._zigzag_step = 0
        self._loop_direction = 1
        self._loop_step = 0
        self._last_signal_bias = 0

    def act(
        self,
        sensor_reading: SensorReading,
        *,
        heading_relative_to_wind: float = pi,
    ) -> Action:
        """Choose the next discrete action from local sensor and wind cues."""

        if sensor_reading.detected:
            self.state = MothState.SURGE
            self.steps_since_detection = 0
            return self._surge_action(sensor_reading, heading_relative_to_wind)

        self.steps_since_detection += 1
        if self.steps_since_detection <= self.config.zigzag_duration:
            if self.state is not MothState.ZIGZAG:
                self._enter_zigzag()
            self.state = MothState.ZIGZAG
            return self._zigzag_action()

        if self.state is not MothState.LOOP:
            self._enter_loop()
        self.state = MothState.LOOP
        return self._loop_action()

    def _surge_action(
        self,
        reading: SensorReading,
        heading_relative_to_wind: float,
    ) -> Action:
        """Move on odor, steering first by bilateral difference then wind."""

        difference = reading.left - reading.right
        if difference > self.config.sensor_deadband:
            self._last_signal_bias = 1
            return Action.TURN_LEFT
        if difference < -self.config.sensor_deadband:
            self._last_signal_bias = -1
            return Action.TURN_RIGHT

        # Equal readings give no steering direction, so use the wind compass.
        upwind_error = _wrap_angle(pi - heading_relative_to_wind)
        if upwind_error > self.config.upwind_tolerance:
            return Action.TURN_LEFT
        if upwind_error < -self.config.upwind_tolerance:
            return Action.TURN_RIGHT
        return Action.FORWARD

    def _enter_zigzag(self) -> None:
        # Start toward the most recent stronger side. Keep the previous direction
        # when the final reading was balanced.
        if self._last_signal_bias:
            self._zigzag_direction = self._last_signal_bias
        self._zigzag_step = 0

    def _zigzag_action(self) -> Action:
        """Alternate short left/right moving arcs near the last odor contact."""

        action = Action.TURN_LEFT if self._zigzag_direction > 0 else Action.TURN_RIGHT
        self._zigzag_step += 1
        if self._zigzag_step >= self.config.zigzag_leg_steps:
            self._zigzag_direction *= -1
            self._zigzag_step = 0
        return action

    def _enter_loop(self) -> None:
        self._loop_direction = self._zigzag_direction
        self._loop_step = 0

    def _loop_action(self) -> Action:
        """Trace a wide polygonal loop by interleaving turns and forward motion."""

        cycle_length = self.config.loop_forward_steps + 1
        turn_now = self._loop_step % cycle_length == 0
        self._loop_step += 1
        if turn_now:
            return Action.SHARP_LEFT if self._loop_direction > 0 else Action.SHARP_RIGHT
        return Action.FORWARD
