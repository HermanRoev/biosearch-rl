from __future__ import annotations

from dataclasses import replace
from math import pi

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gymnasium_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from biosearch.config import SensorConfig, SimulationConfig
from biosearch.environment import Action
from biosearch.evaluation import sample_domain_randomized_config
from biosearch.gym_environment import BioSearchEnv


def fixed_reset_options(
    position: tuple[float, float] = (16.5, 6.0),
    heading: float = pi,
) -> dict[str, object]:
    return {
        "randomize_start": False,
        "agent_position": position,
        "heading": heading,
    }


def test_environment_passes_gymnasium_and_sb3_checkers() -> None:
    environment = BioSearchEnv()

    gymnasium_check_env(environment, skip_render_check=True)
    sb3_check_env(environment, warn=True)

    environment.close()


def test_registered_environment_can_be_created() -> None:
    environment = gym.make("BioSearch-v0")

    observation, _ = environment.reset(seed=4)

    assert environment.observation_space.contains(observation)
    environment.close()


def test_observation_is_local_normalized_and_has_no_source_data() -> None:
    environment = BioSearchEnv(randomize_start=False)

    observation, _ = environment.reset(seed=42)

    assert environment.observation_space.contains(observation)
    assert observation.dtype == np.float32
    forbidden_fragments = ("source", "distance", "plume", "global")
    assert not any(
        fragment in name
        for name in environment.observation_names
        for fragment in forbidden_fragments
    )
    assert environment.observation_names == (
        "left_odor",
        "right_odor",
        "odor_difference",
        "odor_detected",
        "time_since_detection",
        "wind_heading_sin",
        "wind_heading_cos",
        "previous_action",
        "left_odor_moving_average",
        "right_odor_moving_average",
        "front_obstacle_proximity",
        "left_obstacle_proximity",
        "right_obstacle_proximity",
    )


def test_seeded_reset_and_step_are_deterministic() -> None:
    environment = BioSearchEnv()
    first_observation, _ = environment.reset(seed=91)
    first_step = environment.step(Action.TURN_LEFT)

    second_observation, _ = environment.reset(seed=91)
    second_step = environment.step(Action.TURN_LEFT)

    np.testing.assert_array_equal(first_observation, second_observation)
    np.testing.assert_array_equal(first_step[0], second_step[0])
    assert first_step[1:] == second_step[1:]


def test_seeded_domain_randomization_is_deterministic() -> None:
    environment = BioSearchEnv(config_sampler=sample_domain_randomized_config)

    first_observation, _ = environment.reset(seed=91)
    first_config = environment.config
    second_observation, _ = environment.reset(seed=91)
    second_config = environment.config

    assert first_config == second_config
    np.testing.assert_array_equal(first_observation, second_observation)


def test_distance_shaping_is_reward_only_and_positive_for_progress() -> None:
    environment = BioSearchEnv(randomize_start=False)
    environment.reset(seed=5, options=fixed_reset_options())

    _, reward, _, _, info = environment.step(Action.FORWARD)

    assert info["reward_terms"]["distance_progress"] > 0
    assert reward > environment.reward_config.step_penalty
    assert "distance" not in environment.observation_names


def test_disabled_left_sensor_is_zero_in_observation() -> None:
    config = SimulationConfig(sensors=replace(SensorConfig(), disabled_sensor="left"))
    environment = BioSearchEnv(config, randomize_start=False)
    environment.reset(seed=7, options=fixed_reset_options(position=(3.0, 6.0)))

    for _ in range(4):
        observation, _, _, _, _ = environment.step(Action.STILL)

    left_index = environment.observation_names.index("left_odor")
    left_average_index = environment.observation_names.index("left_odor_moving_average")
    assert observation[left_index] == 0.0
    assert observation[left_average_index] == 0.0


def test_front_obstacle_sensor_detects_nearby_boundary() -> None:
    environment = BioSearchEnv(randomize_start=False, obstacle_sensor_range=2.5)

    near_boundary, _ = environment.reset(
        seed=2,
        options=fixed_reset_options(position=(19.5, 6.0), heading=0.0),
    )
    far_from_boundary, _ = environment.reset(
        seed=2,
        options=fixed_reset_options(position=(16.5, 6.0), heading=0.0),
    )

    front_index = environment.observation_names.index("front_obstacle_proximity")
    assert near_boundary[front_index] > 0.8
    assert far_from_boundary[front_index] == pytest.approx(0.0)


def test_step_returns_gymnasium_five_tuple() -> None:
    environment = BioSearchEnv(randomize_start=False)
    environment.reset(seed=1)

    result = environment.step(Action.STILL)

    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert environment.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
