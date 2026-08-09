"""A cheap stochastic odor-puff model.

This module does not implement computational fluid dynamics. It represents odor
as point-like puffs that drift downwind, diffuse laterally, and decay. The model
exists to create intermittent local measurements for control experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biosearch.config import PlumeConfig, WorldConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PuffSnapshot:
    """Read-only copy of plume state for rendering and diagnostics."""

    positions: FloatArray
    intensities: FloatArray
    ages: NDArray[np.int64]


class PuffPlume:
    """Emit, move, diffuse, and decay abstract odor puffs."""

    def __init__(
        self,
        config: PlumeConfig,
        world: WorldConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.world = world
        self.rng = rng
        self.positions = np.empty((0, 2), dtype=np.float64)
        self.intensities = np.empty(0, dtype=np.float64)
        self.ages = np.empty(0, dtype=np.int64)
        self.step_count = 0

        self._downwind = np.array(
            [np.cos(world.wind_direction), np.sin(world.wind_direction)],
            dtype=np.float64,
        )
        self._crosswind = np.array(
            [-self._downwind[1], self._downwind[0]],
            dtype=np.float64,
        )

    def reset(self) -> None:
        """Remove all puffs and restart the emission clock."""

        self.positions = np.empty((0, 2), dtype=np.float64)
        self.intensities = np.empty(0, dtype=np.float64)
        self.ages = np.empty(0, dtype=np.int64)
        self.step_count = 0

    def _emit(self) -> None:
        count = self.config.puffs_per_emission
        source = np.asarray(self.world.source_position, dtype=np.float64)
        # A small crosswind spread prevents all newly emitted puffs overlapping.
        offsets = self.rng.normal(0.0, 0.06, size=count)[:, None] * self._crosswind
        new_positions = source + offsets
        self.positions = np.concatenate((self.positions, new_positions), axis=0)
        self.intensities = np.concatenate(
            (self.intensities, np.full(count, self.config.initial_intensity))
        )
        self.ages = np.concatenate((self.ages, np.zeros(count, dtype=np.int64)))

    def step(self) -> None:
        """Advance the plume by one simulation step."""

        if self.step_count % self.config.emission_interval == 0:
            self._emit()

        count = len(self.positions)
        if count:
            downwind_noise = self.rng.normal(0.0, self.config.longitudinal_jitter, size=count)
            crosswind_noise = self.rng.normal(0.0, self.config.lateral_diffusion, size=count)
            drift = (
                self.world.wind_speed * self._downwind[None, :]
                + downwind_noise[:, None] * self._downwind
                + crosswind_noise[:, None] * self._crosswind
            )
            self.positions += drift
            self.ages += 1
            self.intensities *= 1.0 - self.config.intensity_decay

            margin = 2.0 * self.config.influence_radius
            inside = (
                (self.positions[:, 0] >= -margin)
                & (self.positions[:, 0] <= self.world.width + margin)
                & (self.positions[:, 1] >= -margin)
                & (self.positions[:, 1] <= self.world.height + margin)
            )
            alive = inside & (self.ages <= self.config.max_age) & (self.intensities > 1e-4)
            self.positions = self.positions[alive]
            self.intensities = self.intensities[alive]
            self.ages = self.ages[alive]

            if len(self.positions) > self.config.max_puffs:
                keep_from = len(self.positions) - self.config.max_puffs
                self.positions = self.positions[keep_from:]
                self.intensities = self.intensities[keep_from:]
                self.ages = self.ages[keep_from:]

        self.step_count += 1

    def concentration_at(self, points: FloatArray) -> FloatArray:
        """Return summed local puff influence at one or more ``(x, y)`` points."""

        query = np.asarray(points, dtype=np.float64)
        if query.ndim == 1:
            query = query[None, :]
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("points must have shape (2,) or (n, 2).")
        if len(self.positions) == 0:
            return np.zeros(len(query), dtype=np.float64)

        displacements = query[:, None, :] - self.positions[None, :, :]
        squared_distances = np.sum(displacements * displacements, axis=2)
        variance = self.config.influence_radius**2
        weights = np.exp(-0.5 * squared_distances / variance)
        return weights @ self.intensities

    def snapshot(self) -> PuffSnapshot:
        """Return copies so visual code cannot mutate the simulation."""

        return PuffSnapshot(
            positions=self.positions.copy(),
            intensities=self.intensities.copy(),
            ages=self.ages.copy(),
        )
