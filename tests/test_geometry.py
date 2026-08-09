from __future__ import annotations

from dataclasses import replace

import numpy as np

from biosearch.ablations import ObservationCondition
from biosearch.config import AgentConfig, SimulationConfig
from biosearch.evaluation import ControllerName, SensorCondition
from biosearch.experiments import evaluate_policy_batch, experiment_config
from biosearch.geometry import (
    GeometryCondition,
    GeometryEpisode,
    aggregate_geometry_episodes,
    compare_paired_geometry_outcomes,
    geometry_design_for_seed,
    geometry_episode_setup,
)


def test_geometry_design_is_paired_deterministic_and_decoupled() -> None:
    config = SimulationConfig()
    aligned = geometry_design_for_seed(
        7_000,
        config,
        GeometryCondition.SHIFTED_ALIGNED,
    )
    repeated = geometry_design_for_seed(
        7_000,
        config,
        GeometryCondition.SHIFTED_ALIGNED,
    )
    decoupled = geometry_design_for_seed(
        7_000,
        config,
        GeometryCondition.CROSSWIND_DECOUPLED,
    )

    assert aligned == repeated
    assert aligned.source_position == decoupled.source_position
    assert aligned.start_heading == decoupled.start_heading
    assert aligned.source_position[1] == 3.0
    assert abs(aligned.start_position[1] - 3.0) <= 1.0
    assert abs(decoupled.start_position[1] - 9.0) <= 1.0
    assert aligned.crosswind_separation <= 1.0
    assert 5.0 <= decoupled.crosswind_separation <= 7.0


def test_geometry_design_balances_source_and_start_lane_marginals() -> None:
    config = SimulationConfig()
    aligned = [
        geometry_design_for_seed(seed, config, GeometryCondition.SHIFTED_ALIGNED)
        for seed in range(7_000, 7_050)
    ]
    decoupled = [
        geometry_design_for_seed(seed, config, GeometryCondition.CROSSWIND_DECOUPLED)
        for seed in range(7_000, 7_050)
    ]

    assert sum(row.source_position[1] == 3.0 for row in aligned) == 25
    assert sum(row.source_position[1] == 9.0 for row in aligned) == 25
    assert sum(row.start_position[1] < 6.0 for row in aligned) == 25
    assert sum(row.start_position[1] < 6.0 for row in decoupled) == 25


def test_geometry_episode_setup_changes_source_and_removes_obstacles() -> None:
    config = SimulationConfig()
    setup = geometry_episode_setup(
        7_001,
        config,
        condition=GeometryCondition.CROSSWIND_DECOUPLED,
    )

    assert setup.config.world.source_position == (2.5, 9.0)
    assert setup.config.world.obstacles == ()
    assert setup.position == setup.config.world.agent_start


def test_policy_batch_uses_per_seed_episode_setup() -> None:
    class ForwardModel:
        def predict(
            self,
            observation: np.ndarray,
            *,
            deterministic: bool,
        ) -> tuple[np.ndarray, None]:
            assert deterministic
            return np.zeros(len(observation), dtype=np.int64), None

    base = replace(
        experiment_config(max_steps=1),
        agent=replace(AgentConfig(), max_steps=1),
    )
    runs = evaluate_policy_batch(
        ForwardModel(),  # type: ignore[arg-type]
        controller_name=ControllerName.PPO_NORMAL,
        condition=SensorCondition.NORMAL,
        seeds=(7_000,),
        base_config=base,
        episode_setup_factory=lambda seed, config: geometry_episode_setup(
            seed,
            config,
            condition=GeometryCondition.CROSSWIND_DECOUPLED,
        ),
    )

    design = geometry_design_for_seed(
        7_000,
        base,
        GeometryCondition.CROSSWIND_DECOUPLED,
    )
    assert runs[0].start_position == design.start_position
    expected_distance = np.linalg.norm(runs[0].trajectory[-1] - np.asarray(design.source_position))
    assert runs[0].metrics.final_distance_to_source == expected_distance


def _episode(
    geometry: GeometryCondition,
    observation: ObservationCondition,
    seed: int,
    success: bool,
) -> GeometryEpisode:
    return GeometryEpisode(
        policy="ppo_robust",
        geometry_condition=geometry.value,
        observation_condition=observation.value,
        seed=seed,
        source_y=3.0,
        start_y=3.0 if geometry is GeometryCondition.SHIFTED_ALIGNED else 9.0,
        crosswind_separation=(0.0 if geometry is GeometryCondition.SHIFTED_ALIGNED else 6.0),
        success=success,
        steps=10,
        path_length=2.0,
        collisions=0,
        tortuosity=1.0,
        actual_odor_detection_percentage=20.0,
        final_distance_to_source=0.2 if success else 5.0,
        crosswind_span=7.0,
        closest_crosswind_distance_to_source_lane=0.1,
        crossed_world_centerline=True,
    )


def test_geometry_aggregation_and_paired_comparison() -> None:
    episodes = [
        _episode(GeometryCondition.SHIFTED_ALIGNED, ObservationCondition.UNMASKED, 1, True),
        _episode(GeometryCondition.SHIFTED_ALIGNED, ObservationCondition.UNMASKED, 2, True),
        _episode(
            GeometryCondition.CROSSWIND_DECOUPLED,
            ObservationCondition.UNMASKED,
            1,
            False,
        ),
        _episode(
            GeometryCondition.CROSSWIND_DECOUPLED,
            ObservationCondition.UNMASKED,
            2,
            True,
        ),
    ]

    summaries = aggregate_geometry_episodes(episodes)
    comparisons = compare_paired_geometry_outcomes(episodes)

    assert [summary.success_rate for summary in summaries] == [50.0, 100.0]
    assert comparisons[0].reference_only_succeeds == 1
    assert comparisons[0].success_rate_change_percentage_points == -50.0
