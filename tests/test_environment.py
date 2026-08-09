from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from biosearch.config import AgentConfig, SimulationConfig, WorldConfig
from biosearch.environment import Action, BioSearchSimulation


def obstacle_free_config(**agent_changes: float | int) -> SimulationConfig:
    return SimulationConfig(
        world=WorldConfig(obstacles=()),
        agent=replace(AgentConfig(), **agent_changes),
    )


def test_forward_action_moves_in_heading_direction() -> None:
    simulation = BioSearchSimulation(obstacle_free_config(), seed=3)
    simulation.reset(seed=3, agent_position=(10.0, 6.0), heading=0.0)

    simulation.step(Action.FORWARD)

    np.testing.assert_allclose(
        simulation.position,
        [10.0 + simulation.config.agent.forward_speed, 6.0],
        atol=1e-12,
    )


def test_sharp_turn_rotates_without_translating() -> None:
    simulation = BioSearchSimulation(obstacle_free_config(), seed=3)
    start = simulation.position.copy()

    simulation.step(Action.SHARP_LEFT)

    np.testing.assert_allclose(simulation.position, start)
    assert simulation.heading == pytest.approx(
        simulation.config.world.agent_start_heading + simulation.config.agent.sharp_turn - 2 * np.pi
    )


def test_boundary_collision_blocks_motion_and_is_counted() -> None:
    config = obstacle_free_config(forward_speed=0.4)
    simulation = BioSearchSimulation(config, seed=4)
    start = (config.agent.radius + 0.05, 4.0)
    simulation.reset(seed=4, agent_position=start, heading=np.pi)

    result = simulation.step(Action.FORWARD)

    assert result.collision
    assert simulation.collision_count == 1
    np.testing.assert_allclose(simulation.position, start)


def test_reaching_source_terminates_episode() -> None:
    world = WorldConfig(
        source_position=(5.0, 5.0),
        source_radius=0.5,
        agent_start=(5.6, 5.0),
        agent_start_heading=np.pi,
        obstacles=(),
    )
    config = SimulationConfig(world=world, agent=AgentConfig(forward_speed=0.2))
    simulation = BioSearchSimulation(config, seed=5)

    result = simulation.step(Action.FORWARD)

    assert result.success
    assert result.terminated
    assert not result.truncated


def test_time_limit_truncates_episode() -> None:
    config = obstacle_free_config(max_steps=2)
    simulation = BioSearchSimulation(config, seed=6)

    simulation.step(Action.STILL)
    result = simulation.step(Action.STILL)

    assert result.truncated
    with pytest.raises(RuntimeError, match="finished simulation"):
        simulation.step(Action.STILL)


def test_seeded_simulations_match() -> None:
    first = BioSearchSimulation(seed=99)
    second = BioSearchSimulation(seed=99)
    actions = [Action.STILL, Action.TURN_LEFT, Action.FORWARD] * 15

    for action in actions:
        first_result = first.step(action)
        second_result = second.step(action)
        np.testing.assert_allclose(first.plume.positions, second.plume.positions)
        assert first_result.sensor_reading.left == second_result.sensor_reading.left
        assert first_result.sensor_reading.right == second_result.sensor_reading.right


def test_episode_histories_align_with_steps() -> None:
    simulation = BioSearchSimulation(seed=12)
    actions = [Action.STILL, Action.SHARP_LEFT, Action.FORWARD]

    for action in actions:
        simulation.step(action)

    assert len(simulation.trajectory) == len(actions) + 1
    assert len(simulation.detection_history) == len(actions) + 1
    assert len(simulation.sensor_history) == len(actions) + 1
    assert simulation.action_history == actions
    assert len(simulation.collision_history) == len(actions)
