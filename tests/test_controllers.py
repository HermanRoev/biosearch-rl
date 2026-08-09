from math import pi

import numpy as np

from biosearch.controllers import (
    MothController,
    MothControllerConfig,
    MothState,
    RandomController,
)
from biosearch.environment import Action
from biosearch.sensors import SensorReading


def reading(left: float, right: float, *, detected: bool) -> SensorReading:
    return SensorReading(
        left=left,
        right=right,
        detected=detected,
        left_position=np.zeros(2),
        right_position=np.zeros(2),
    )


def test_random_controller_is_seeded() -> None:
    first = RandomController(seed=123)
    second = RandomController(seed=123)

    first_actions = [first.act() for _ in range(30)]
    second_actions = [second.act() for _ in range(30)]

    assert first_actions == second_actions
    assert all(isinstance(action, Action) for action in first_actions)


def test_moth_controller_starts_in_loop_without_prior_odor() -> None:
    controller = MothController()

    action = controller.act(reading(0.0, 0.0, detected=False))

    assert controller.state is MothState.LOOP
    assert action is Action.SHARP_LEFT


def test_odor_enters_surge_and_steers_toward_stronger_sensor() -> None:
    controller = MothController()

    action = controller.act(
        reading(0.7, 0.2, detected=True),
        heading_relative_to_wind=pi,
    )

    assert controller.state is MothState.SURGE
    assert action is Action.TURN_LEFT


def test_balanced_odor_uses_wind_to_orient_upwind() -> None:
    controller = MothController()

    action = controller.act(
        reading(0.5, 0.5, detected=True),
        heading_relative_to_wind=pi / 2,
    )

    assert controller.state is MothState.SURGE
    assert action is Action.TURN_LEFT


def test_odor_loss_transitions_from_zigzag_to_loop() -> None:
    controller = MothController(
        MothControllerConfig(
            zigzag_duration=3,
            zigzag_leg_steps=1,
            loop_forward_steps=2,
        )
    )
    controller.act(reading(0.4, 0.1, detected=True))

    first_search_action = controller.act(reading(0.0, 0.0, detected=False))
    assert controller.state is MothState.ZIGZAG
    assert first_search_action is Action.TURN_LEFT
    assert controller.act(reading(0.0, 0.0, detected=False)) is Action.TURN_RIGHT
    controller.act(reading(0.0, 0.0, detected=False))

    loop_action = controller.act(reading(0.0, 0.0, detected=False))
    assert controller.state is MothState.LOOP
    assert loop_action is Action.SHARP_RIGHT


def test_new_detection_interrupts_loop() -> None:
    controller = MothController()
    controller.act(reading(0.0, 0.0, detected=False))

    action = controller.act(reading(0.3, 0.3, detected=True))

    assert controller.state is MothState.SURGE
    assert action is Action.FORWARD
