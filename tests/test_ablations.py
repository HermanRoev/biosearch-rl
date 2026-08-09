from __future__ import annotations

import numpy as np
import pytest

from biosearch.ablations import (
    TIME_SINCE_DETECTION_INDEX,
    ZEROED_ODOR_INDICES,
    ZEROED_WIND_INDICES,
    AblationEpisode,
    ObservationCondition,
    PairedAblationComparison,
    aggregate_ablation_episodes,
    apply_observation_condition,
    compare_paired_ablation_outcomes,
    holm_adjusted_p_values,
    make_observation_transform,
    wind_rotation_for_seed,
)
from biosearch.evaluation import ControllerName, SensorCondition
from biosearch.experiments import evaluate_policy_batch, experiment_config
from biosearch.gym_environment import BioSearchEnv


def test_odor_blind_condition_masks_only_odor_information() -> None:
    observation = np.linspace(
        0.05,
        0.95,
        len(BioSearchEnv.observation_names),
        dtype=np.float32,
    )

    transformed = apply_observation_condition(
        observation,
        ObservationCondition.ODOR_BLIND,
    )

    expected = observation.copy()
    expected[list(ZEROED_ODOR_INDICES)] = 0.0
    expected[TIME_SINCE_DETECTION_INDEX] = 1.0
    np.testing.assert_array_equal(transformed, expected)
    np.testing.assert_array_equal(
        observation,
        np.linspace(
            0.05,
            0.95,
            len(BioSearchEnv.observation_names),
            dtype=np.float32,
        ),
    )


def test_unmasked_condition_returns_an_independent_copy() -> None:
    observation = np.ones(len(BioSearchEnv.observation_names), dtype=np.float32)

    transformed = apply_observation_condition(
        observation,
        ObservationCondition.UNMASKED,
    )

    np.testing.assert_array_equal(transformed, observation)
    assert transformed is not observation


@pytest.mark.parametrize(
    ("condition", "mask_odor", "mask_wind"),
    (
        (ObservationCondition.ODOR_BLIND, True, False),
        (ObservationCondition.WIND_BLIND, False, True),
        (ObservationCondition.ODOR_WIND_BLIND, True, True),
    ),
)
def test_cue_masks_remove_only_selected_feature_groups(
    condition: ObservationCondition,
    mask_odor: bool,
    mask_wind: bool,
) -> None:
    observation = np.linspace(0.05, 0.95, 13, dtype=np.float32)

    transformed = apply_observation_condition(observation, condition)

    expected = observation.copy()
    if mask_odor:
        expected[list(ZEROED_ODOR_INDICES)] = 0.0
        expected[TIME_SINCE_DETECTION_INDEX] = 1.0
    if mask_wind:
        expected[list(ZEROED_WIND_INDICES)] = 0.0
    np.testing.assert_array_equal(transformed, expected)


def test_observation_condition_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="observation must be"):
        apply_observation_condition(
            np.ones(3, dtype=np.float32),
            ObservationCondition.ODOR_BLIND,
        )


@pytest.mark.parametrize(
    ("condition", "odor_hidden"),
    (
        (ObservationCondition.WIND_ROTATED, False),
        (ObservationCondition.ODOR_BLIND_WIND_ROTATED, True),
    ),
)
def test_rotated_wind_is_valid_and_changes_only_selected_cues(
    condition: ObservationCondition,
    odor_hidden: bool,
) -> None:
    observation = np.linspace(0.05, 0.95, 13, dtype=np.float32)
    wind_sin_index, wind_cos_index = ZEROED_WIND_INDICES
    observation[wind_sin_index] = 0.6
    observation[wind_cos_index] = 0.8

    transformed = apply_observation_condition(
        observation,
        condition,
        wind_rotation_radians=np.pi / 2.0,
    )

    assert transformed[wind_sin_index] == pytest.approx(0.8)
    assert transformed[wind_cos_index] == pytest.approx(-0.6)
    assert np.hypot(transformed[wind_sin_index], transformed[wind_cos_index]) == pytest.approx(1.0)
    expected = observation.copy()
    expected[wind_sin_index] = 0.8
    expected[wind_cos_index] = -0.6
    if odor_hidden:
        expected[list(ZEROED_ODOR_INDICES)] = 0.0
        expected[TIME_SINCE_DETECTION_INDEX] = 1.0
    np.testing.assert_allclose(transformed, expected)


def test_seeded_wind_rotation_is_balanced_and_temporally_consistent() -> None:
    rotations = [wind_rotation_for_seed(seed) for seed in range(6_000, 6_050)]

    assert rotations.count(-np.pi / 2.0) == 25
    assert rotations.count(np.pi / 2.0) == 25
    transform = make_observation_transform(
        6_001,
        condition=ObservationCondition.WIND_ROTATED,
    )
    observation = np.zeros(13, dtype=np.float32)
    wind_sin_index, wind_cos_index = ZEROED_WIND_INDICES
    observation[wind_cos_index] = 1.0
    first = transform(observation)
    second = transform(observation)
    np.testing.assert_array_equal(first, second)
    assert first[wind_sin_index] == pytest.approx(1.0)


def _episode(
    condition: ObservationCondition,
    seed: int,
    success: bool,
) -> AblationEpisode:
    return AblationEpisode(
        policy="ppo_normal",
        observation_condition=condition.value,
        physical_condition="normal",
        seed=seed,
        success=success,
        steps=10 if success else 20,
        path_length=2.0,
        collisions=0,
        tortuosity=1.2,
        actual_odor_detection_percentage=30.0,
        final_distance_to_source=0.2 if success else 3.0,
    )


def test_ablation_aggregation_and_paired_comparison() -> None:
    episodes = [
        _episode(ObservationCondition.UNMASKED, 1, True),
        _episode(ObservationCondition.UNMASKED, 2, True),
        _episode(ObservationCondition.ODOR_BLIND, 1, False),
        _episode(ObservationCondition.ODOR_BLIND, 2, True),
    ]

    summaries = aggregate_ablation_episodes(episodes)
    comparisons = compare_paired_ablation_outcomes(episodes)

    assert [row.success_rate for row in summaries] == [50.0, 100.0]
    assert comparisons[0].both_succeed == 1
    assert comparisons[0].reference_only_succeeds == 1
    assert comparisons[0].comparison_only_succeeds == 0
    assert comparisons[0].success_rate_change_percentage_points == -50.0
    assert comparisons[0].exact_mcnemar_p_value == 1.0


def test_policy_batch_applies_transform_only_to_model_input() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.inputs: list[np.ndarray] = []

        def predict(
            self,
            observation: np.ndarray,
            *,
            deterministic: bool,
        ) -> tuple[np.ndarray, None]:
            assert deterministic
            self.inputs.append(observation.copy())
            return np.zeros(len(observation), dtype=np.int64), None

    transform_calls: list[np.ndarray] = []

    def transform(observation: np.ndarray) -> np.ndarray:
        transform_calls.append(observation.copy())
        return np.full_like(observation, 0.5)

    model = RecordingModel()
    runs = evaluate_policy_batch(
        model,  # type: ignore[arg-type]
        controller_name=ControllerName.PPO_NORMAL,
        condition=SensorCondition.NORMAL,
        seeds=(1, 2),
        base_config=experiment_config(max_steps=1),
        observation_transform=transform,
    )

    assert len(transform_calls) == 4
    assert len(model.inputs) == 1
    np.testing.assert_array_equal(model.inputs[0], np.full((2, 13), 0.5))
    assert all(run.sensor_values.shape == (2, 2) for run in runs)


def test_policy_batch_supports_one_transform_per_seed() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.inputs: list[np.ndarray] = []

        def predict(
            self,
            observation: np.ndarray,
            *,
            deterministic: bool,
        ) -> tuple[np.ndarray, None]:
            assert deterministic
            self.inputs.append(observation.copy())
            return np.zeros(len(observation), dtype=np.int64), None

    def factory(seed: int):
        def transform(observation: np.ndarray) -> np.ndarray:
            return np.full_like(observation, seed / 10.0)

        return transform

    model = RecordingModel()
    evaluate_policy_batch(
        model,  # type: ignore[arg-type]
        controller_name=ControllerName.PPO_NORMAL,
        condition=SensorCondition.NORMAL,
        seeds=(1, 2),
        base_config=experiment_config(max_steps=1),
        observation_transform_factory=factory,
    )

    assert len(model.inputs) == 1
    np.testing.assert_array_equal(
        model.inputs[0],
        np.vstack(
            (
                np.full(13, 0.1, dtype=np.float32),
                np.full(13, 0.2, dtype=np.float32),
            )
        ),
    )


def test_holm_adjustment_is_monotonic_within_policy() -> None:
    comparisons = [
        PairedAblationComparison(
            policy="ppo_normal",
            reference_condition="unmasked",
            comparison_condition=condition,
            episodes=50,
            both_succeed=1,
            reference_only_succeeds=1,
            comparison_only_succeeds=0,
            neither_succeeds=48,
            success_rate_change_percentage_points=-2.0,
            exact_mcnemar_p_value=p_value,
        )
        for condition, p_value in (
            ("first", 0.01),
            ("second", 0.04),
            ("third", 0.03),
        )
    ]

    adjusted = holm_adjusted_p_values(comparisons)

    assert adjusted[("ppo_normal", "first")] == pytest.approx(0.03)
    assert adjusted[("ppo_normal", "third")] == pytest.approx(0.06)
    assert adjusted[("ppo_normal", "second")] == pytest.approx(0.06)
