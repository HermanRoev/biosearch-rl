"""Seeded random-action baseline used by the Milestone 1 demo."""

from __future__ import annotations

import numpy as np

from biosearch.environment import Action
from biosearch.sensors import SensorReading


class RandomController:
    """Choose each discrete action uniformly and reproducibly."""

    state_name = "RANDOM"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Restart the controller's random sequence."""

        self._rng = np.random.default_rng(seed)

    def act(self, sensor_reading: SensorReading | None = None) -> Action:
        """Return a random action; the reading is accepted for a common API."""

        del sensor_reading
        return Action(int(self._rng.integers(0, len(Action))))
