"""Observation ablations used to audit learned-policy behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import comb, cos, isfinite, pi, sin
from statistics import fmean
from typing import Any

import numpy as np
from numpy.typing import NDArray

from biosearch.evaluation import EpisodeRun, success_rate_interval
from biosearch.gym_environment import BioSearchEnv

Observation = NDArray[np.float32]


class ObservationCondition(StrEnum):
    """Policy-input conditions for inference-only diagnostics."""

    UNMASKED = "unmasked"
    ODOR_BLIND = "odor_blind"
    WIND_BLIND = "wind_blind"
    ODOR_WIND_BLIND = "odor_wind_blind"
    WIND_ROTATED = "wind_rotated"
    ODOR_BLIND_WIND_ROTATED = "odor_blind_wind_rotated"


@dataclass(frozen=True)
class AblationEpisode:
    """One policy episode with its policy-input condition."""

    policy: str
    observation_condition: str
    physical_condition: str
    seed: int
    success: bool
    steps: int
    path_length: float
    collisions: int
    tortuosity: float
    actual_odor_detection_percentage: float
    final_distance_to_source: float

    def as_dict(self) -> dict[str, Any]:
        """Return a CSV-friendly field mapping."""

        return asdict(self)


@dataclass(frozen=True)
class AblationSummary:
    """Aggregate results for one policy-input condition."""

    policy: str
    observation_condition: str
    episodes: int
    success_rate: float
    success_ci95_low: float
    success_ci95_high: float
    mean_success_steps: float
    mean_path_length: float
    mean_actual_odor_detection_percentage: float

    def as_dict(self) -> dict[str, Any]:
        """Return a CSV-friendly field mapping."""

        return asdict(self)


@dataclass(frozen=True)
class PairedAblationComparison:
    """Paired success outcomes for two policy-input conditions."""

    policy: str
    reference_condition: str
    comparison_condition: str
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


ZEROED_ODOR_FEATURES = (
    "left_odor",
    "right_odor",
    "odor_difference",
    "odor_detected",
    "left_odor_moving_average",
    "right_odor_moving_average",
)
ZEROED_ODOR_INDICES = tuple(
    BioSearchEnv.observation_names.index(name) for name in ZEROED_ODOR_FEATURES
)
TIME_SINCE_DETECTION_INDEX = BioSearchEnv.observation_names.index("time_since_detection")
ZEROED_WIND_FEATURES = (
    "wind_heading_sin",
    "wind_heading_cos",
)
ZEROED_WIND_INDICES = tuple(
    BioSearchEnv.observation_names.index(name) for name in ZEROED_WIND_FEATURES
)


def apply_observation_condition(
    observation: Observation,
    condition: ObservationCondition | str,
    *,
    wind_rotation_radians: float = pi / 2.0,
) -> Observation:
    """Return a policy observation under an inference-only ablation.

    Odor masking represents permanent odor absence: current and historical odor
    features are zero, while normalized time since detection is one. Wind
    masking sets both wind-heading components to zero so they contain no angle.
    Wind rotation supplies a valid but incorrect direction by rotating the
    sine/cosine pair. Obstacle rays and previous action are unchanged. The
    underlying simulator, plume, sensors, reward, and recorded detections remain
    intact.
    """

    selected = ObservationCondition(condition)
    transformed = np.asarray(observation, dtype=np.float32).copy()
    if transformed.shape != (len(BioSearchEnv.observation_names),):
        raise ValueError(
            "observation must be one BioSearchEnv observation with shape "
            f"({len(BioSearchEnv.observation_names)},)."
        )
    if selected in (
        ObservationCondition.ODOR_BLIND,
        ObservationCondition.ODOR_WIND_BLIND,
        ObservationCondition.ODOR_BLIND_WIND_ROTATED,
    ):
        transformed[list(ZEROED_ODOR_INDICES)] = 0.0
        transformed[TIME_SINCE_DETECTION_INDEX] = 1.0
    if selected in (
        ObservationCondition.WIND_BLIND,
        ObservationCondition.ODOR_WIND_BLIND,
    ):
        transformed[list(ZEROED_WIND_INDICES)] = 0.0
    if selected in (
        ObservationCondition.WIND_ROTATED,
        ObservationCondition.ODOR_BLIND_WIND_ROTATED,
    ):
        if not isfinite(wind_rotation_radians):
            raise ValueError("wind_rotation_radians must be finite.")
        wind_sin_index, wind_cos_index = ZEROED_WIND_INDICES
        wind_sin = float(transformed[wind_sin_index])
        wind_cos = float(transformed[wind_cos_index])
        rotation_sin = sin(wind_rotation_radians)
        rotation_cos = cos(wind_rotation_radians)
        transformed[wind_sin_index] = wind_sin * rotation_cos + wind_cos * rotation_sin
        transformed[wind_cos_index] = wind_cos * rotation_cos - wind_sin * rotation_sin
    return transformed


def wind_rotation_for_seed(seed: int) -> float:
    """Return a balanced deterministic -90/+90 degree rotation by seed parity."""

    return -pi / 2.0 if seed % 2 == 0 else pi / 2.0


def make_observation_transform(
    seed: int,
    *,
    condition: ObservationCondition | str,
) -> Callable[[Observation], Observation]:
    """Build one temporally consistent inference transform for an episode seed."""

    selected = ObservationCondition(condition)
    rotation = wind_rotation_for_seed(seed)

    def transform(observation: Observation) -> Observation:
        return apply_observation_condition(
            observation,
            selected,
            wind_rotation_radians=rotation,
        )

    return transform


def ablation_episode_from_run(
    run: EpisodeRun,
    observation_condition: ObservationCondition,
) -> AblationEpisode:
    """Attach an inference condition to an ordinary episode result."""

    metrics = run.metrics
    return AblationEpisode(
        policy=metrics.controller,
        observation_condition=observation_condition.value,
        physical_condition=metrics.condition,
        seed=metrics.seed,
        success=metrics.success,
        steps=metrics.steps,
        path_length=metrics.path_length,
        collisions=metrics.collisions,
        tortuosity=metrics.tortuosity,
        actual_odor_detection_percentage=metrics.odor_detection_percentage,
        final_distance_to_source=metrics.final_distance_to_source,
    )


def aggregate_ablation_episodes(
    episodes: list[AblationEpisode],
) -> list[AblationSummary]:
    """Aggregate observation-ablation rows by policy and input condition."""

    grouped: dict[tuple[str, str], list[AblationEpisode]] = {}
    for episode in episodes:
        key = (episode.policy, episode.observation_condition)
        grouped.setdefault(key, []).append(episode)
    summaries: list[AblationSummary] = []
    for (policy, condition), rows in sorted(grouped.items()):
        successes = sum(row.success for row in rows)
        ci_low, ci_high = success_rate_interval(successes, len(rows))
        successful_steps = [row.steps for row in rows if row.success]
        summaries.append(
            AblationSummary(
                policy=policy,
                observation_condition=condition,
                episodes=len(rows),
                success_rate=100.0 * successes / len(rows),
                success_ci95_low=ci_low,
                success_ci95_high=ci_high,
                mean_success_steps=(fmean(successful_steps) if successful_steps else float("nan")),
                mean_path_length=fmean(row.path_length for row in rows),
                mean_actual_odor_detection_percentage=fmean(
                    row.actual_odor_detection_percentage for row in rows
                ),
            )
        )
    return summaries


def compare_paired_ablation_outcomes(
    episodes: list[AblationEpisode],
    *,
    reference_condition: ObservationCondition = ObservationCondition.UNMASKED,
    comparison_condition: ObservationCondition = ObservationCondition.ODOR_BLIND,
) -> list[PairedAblationComparison]:
    """Compare paired binary outcomes with an exact McNemar test."""

    policies = sorted({episode.policy for episode in episodes})
    comparisons: list[PairedAblationComparison] = []
    for policy in policies:
        rows = [episode for episode in episodes if episode.policy == policy]
        reference = {
            row.seed: row.success
            for row in rows
            if row.observation_condition == reference_condition.value
        }
        comparison = {
            row.seed: row.success
            for row in rows
            if row.observation_condition == comparison_condition.value
        }
        if not reference or not comparison:
            raise ValueError("Both paired ablation conditions must be present.")
        if set(reference) != set(comparison):
            raise ValueError("Paired ablation conditions must use identical seeds.")
        outcome_counts = {(reference[seed], comparison[seed]): 0 for seed in reference}
        for seed in reference:
            outcome = (reference[seed], comparison[seed])
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        both = outcome_counts.get((True, True), 0)
        reference_only = outcome_counts.get((True, False), 0)
        comparison_only = outcome_counts.get((False, True), 0)
        neither = outcome_counts.get((False, False), 0)
        episodes_count = len(reference)
        comparisons.append(
            PairedAblationComparison(
                policy=policy,
                reference_condition=reference_condition.value,
                comparison_condition=comparison_condition.value,
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


def holm_adjusted_p_values(
    comparisons: list[PairedAblationComparison],
) -> dict[tuple[str, str], float]:
    """Holm-adjust paired tests independently within each policy."""

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
            adjusted[(policy, row.comparison_condition)] = running_max
    return adjusted


def exact_mcnemar_p_value(first_only: int, second_only: int) -> float:
    """Return the two-sided exact binomial McNemar p-value."""

    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        comb(discordant, value) for value in range(min(first_only, second_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)
