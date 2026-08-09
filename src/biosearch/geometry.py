"""Deterministic geometry-shift designs and paired outcome analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import pi
from statistics import fmean
from typing import Any

import numpy as np
from numpy.typing import NDArray

from biosearch.ablations import ObservationCondition, exact_mcnemar_p_value
from biosearch.config import SimulationConfig
from biosearch.evaluation import EpisodeRun, success_rate_interval
from biosearch.experiments import EpisodeSetup


class GeometryCondition(StrEnum):
    """Source/start relationships in the Phase 5.4 diagnostic."""

    SHIFTED_ALIGNED = "shifted_aligned"
    CROSSWIND_DECOUPLED = "crosswind_decoupled"


@dataclass(frozen=True)
class GeometryDesign:
    """Fixed source and initial pose for one seed and geometry condition."""

    condition: GeometryCondition
    source_position: tuple[float, float]
    start_position: tuple[float, float]
    start_heading: float

    @property
    def crosswind_separation(self) -> float:
        """Return absolute source/start separation perpendicular to the wind."""

        return abs(self.start_position[1] - self.source_position[1])


@dataclass(frozen=True)
class GeometryEpisode:
    """One frozen-policy episode under a source/start geometry condition."""

    policy: str
    geometry_condition: str
    observation_condition: str
    seed: int
    source_y: float
    start_y: float
    crosswind_separation: float
    success: bool
    steps: int
    path_length: float
    collisions: int
    tortuosity: float
    actual_odor_detection_percentage: float
    final_distance_to_source: float
    crosswind_span: float
    closest_crosswind_distance_to_source_lane: float
    crossed_world_centerline: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a CSV-friendly field mapping."""

        return asdict(self)


@dataclass(frozen=True)
class GeometrySummary:
    """Aggregate results for one policy, geometry, and observation condition."""

    policy: str
    geometry_condition: str
    observation_condition: str
    episodes: int
    success_rate: float
    success_ci95_low: float
    success_ci95_high: float
    mean_success_steps: float
    mean_path_length: float
    mean_crosswind_separation: float
    mean_actual_odor_detection_percentage: float

    def as_dict(self) -> dict[str, Any]:
        """Return a CSV-friendly field mapping."""

        return asdict(self)


@dataclass(frozen=True)
class PairedGeometryComparison:
    """Paired success outcomes for aligned and decoupled source/start geometry."""

    policy: str
    observation_condition: str
    reference_geometry: str
    comparison_geometry: str
    episodes: int
    both_succeed: int
    reference_only_succeeds: int
    comparison_only_succeeds: int
    neither_succeeds: int
    success_rate_change_percentage_points: float
    exact_mcnemar_p_value: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable field mapping."""

        return asdict(self)


@dataclass(frozen=True)
class GeometryTrajectory:
    """Trajectory plus design labels for the post-hoc casting audit."""

    policy: str
    geometry_condition: str
    observation_condition: str
    seed: int
    source_position: tuple[float, float]
    start_position: tuple[float, float]
    success: bool
    trajectory: NDArray[np.float64]


def geometry_design_for_seed(
    seed: int,
    base_config: SimulationConfig,
    condition: GeometryCondition | str,
) -> GeometryDesign:
    """Return a reproducible shifted-lane design with matched lane marginals.

    Sources occupy the lower and upper quarter-height lanes in equal numbers.
    In the aligned condition, the start uses the source lane. In the decoupled
    condition it uses the opposite lane. The same crosswind jitter and heading
    are used for a seed in both conditions.
    """

    selected = GeometryCondition(condition)
    lower_lane = 0.25 * base_config.world.height
    upper_lane = 0.75 * base_config.world.height
    source_y = lower_lane if seed % 2 == 0 else upper_lane
    opposite_y = upper_lane if seed % 2 == 0 else lower_lane
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2_026, 504]))
    crosswind_jitter = float(rng.uniform(-1.0, 1.0))
    start_lane = source_y if selected is GeometryCondition.SHIFTED_ALIGNED else opposite_y
    source_position = (base_config.world.source_position[0], source_y)
    start_position = (
        base_config.world.agent_start[0],
        start_lane + crosswind_jitter,
    )
    start_heading = float(base_config.world.agent_start_heading + rng.uniform(-pi / 4.0, pi / 4.0))
    return GeometryDesign(
        condition=selected,
        source_position=source_position,
        start_position=start_position,
        start_heading=start_heading,
    )


def geometry_episode_setup(
    seed: int,
    base_config: SimulationConfig,
    *,
    condition: GeometryCondition | str,
) -> EpisodeSetup:
    """Build an open-world episode setup for one geometry condition."""

    design = geometry_design_for_seed(seed, base_config, condition)
    world = replace(
        base_config.world,
        source_position=design.source_position,
        agent_start=design.start_position,
        agent_start_heading=design.start_heading,
        obstacles=(),
    )
    return EpisodeSetup(
        config=replace(base_config, world=world),
        position=design.start_position,
        heading=design.start_heading,
    )


def geometry_episode_from_run(
    run: EpisodeRun,
    design: GeometryDesign,
    observation_condition: ObservationCondition,
) -> GeometryEpisode:
    """Attach the planned geometry and input condition to episode metrics."""

    metrics = run.metrics
    trajectory_y = run.trajectory[:, 1]
    source_y = design.source_position[1]
    world_centerline = 6.0
    return GeometryEpisode(
        policy=metrics.controller,
        geometry_condition=design.condition.value,
        observation_condition=observation_condition.value,
        seed=metrics.seed,
        source_y=design.source_position[1],
        start_y=design.start_position[1],
        crosswind_separation=design.crosswind_separation,
        success=metrics.success,
        steps=metrics.steps,
        path_length=metrics.path_length,
        collisions=metrics.collisions,
        tortuosity=metrics.tortuosity,
        actual_odor_detection_percentage=metrics.odor_detection_percentage,
        final_distance_to_source=metrics.final_distance_to_source,
        crosswind_span=float(np.ptp(trajectory_y)),
        closest_crosswind_distance_to_source_lane=float(np.min(np.abs(trajectory_y - source_y))),
        crossed_world_centerline=bool(
            np.min(trajectory_y) <= world_centerline <= np.max(trajectory_y)
        ),
    )


def geometry_trajectory_from_run(
    run: EpisodeRun,
    design: GeometryDesign,
    observation_condition: ObservationCondition,
) -> GeometryTrajectory:
    """Attach design labels to a recorded trajectory for visual audit."""

    return GeometryTrajectory(
        policy=run.metrics.controller,
        geometry_condition=design.condition.value,
        observation_condition=observation_condition.value,
        seed=run.metrics.seed,
        source_position=design.source_position,
        start_position=design.start_position,
        success=run.metrics.success,
        trajectory=run.trajectory,
    )


def aggregate_geometry_episodes(
    episodes: list[GeometryEpisode],
) -> list[GeometrySummary]:
    """Aggregate geometry-diagnostic rows by policy, geometry, and input."""

    grouped: dict[tuple[str, str, str], list[GeometryEpisode]] = {}
    for episode in episodes:
        key = (
            episode.policy,
            episode.geometry_condition,
            episode.observation_condition,
        )
        grouped.setdefault(key, []).append(episode)
    summaries: list[GeometrySummary] = []
    for (policy, geometry, observation), rows in sorted(grouped.items()):
        successes = sum(row.success for row in rows)
        ci_low, ci_high = success_rate_interval(successes, len(rows))
        successful_steps = [row.steps for row in rows if row.success]
        summaries.append(
            GeometrySummary(
                policy=policy,
                geometry_condition=geometry,
                observation_condition=observation,
                episodes=len(rows),
                success_rate=100.0 * successes / len(rows),
                success_ci95_low=ci_low,
                success_ci95_high=ci_high,
                mean_success_steps=(fmean(successful_steps) if successful_steps else float("nan")),
                mean_path_length=fmean(row.path_length for row in rows),
                mean_crosswind_separation=fmean(row.crosswind_separation for row in rows),
                mean_actual_odor_detection_percentage=fmean(
                    row.actual_odor_detection_percentage for row in rows
                ),
            )
        )
    return summaries


def compare_paired_geometry_outcomes(
    episodes: list[GeometryEpisode],
    *,
    reference_geometry: GeometryCondition = GeometryCondition.SHIFTED_ALIGNED,
    comparison_geometry: GeometryCondition = GeometryCondition.CROSSWIND_DECOUPLED,
) -> list[PairedGeometryComparison]:
    """Compare aligned and decoupled binary outcomes with exact McNemar tests."""

    groups = sorted({(episode.policy, episode.observation_condition) for episode in episodes})
    comparisons: list[PairedGeometryComparison] = []
    for policy, observation in groups:
        rows = [
            episode
            for episode in episodes
            if episode.policy == policy and episode.observation_condition == observation
        ]
        reference = {
            row.seed: row.success
            for row in rows
            if row.geometry_condition == reference_geometry.value
        }
        comparison = {
            row.seed: row.success
            for row in rows
            if row.geometry_condition == comparison_geometry.value
        }
        if not reference or not comparison:
            raise ValueError("Both paired geometry conditions must be present.")
        if set(reference) != set(comparison):
            raise ValueError("Paired geometry conditions must use identical seeds.")
        both = sum(reference[seed] and comparison[seed] for seed in reference)
        reference_only = sum(reference[seed] and not comparison[seed] for seed in reference)
        comparison_only = sum(not reference[seed] and comparison[seed] for seed in reference)
        neither = len(reference) - both - reference_only - comparison_only
        episodes_count = len(reference)
        comparisons.append(
            PairedGeometryComparison(
                policy=policy,
                observation_condition=observation,
                reference_geometry=reference_geometry.value,
                comparison_geometry=comparison_geometry.value,
                episodes=episodes_count,
                both_succeed=both,
                reference_only_succeeds=reference_only,
                comparison_only_succeeds=comparison_only,
                neither_succeeds=neither,
                success_rate_change_percentage_points=(
                    100.0 * (both + comparison_only) / episodes_count
                    - 100.0 * (both + reference_only) / episodes_count
                ),
                exact_mcnemar_p_value=exact_mcnemar_p_value(
                    reference_only,
                    comparison_only,
                ),
            )
        )
    return comparisons


def holm_adjusted_geometry_p_values(
    comparisons: list[PairedGeometryComparison],
) -> dict[tuple[str, str], float]:
    """Holm-adjust the two geometry contrasts independently within policy."""

    adjusted: dict[tuple[str, str], float] = {}
    policies = sorted({comparison.policy for comparison in comparisons})
    for policy in policies:
        rows = sorted(
            (row for row in comparisons if row.policy == policy),
            key=lambda row: row.exact_mcnemar_p_value,
        )
        running_max = 0.0
        tests = len(rows)
        for index, row in enumerate(rows):
            candidate = min(1.0, (tests - index) * row.exact_mcnemar_p_value)
            running_max = max(running_max, candidate)
            adjusted[(policy, row.observation_condition)] = running_max
    return adjusted
