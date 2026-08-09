import numpy as np
import pytest

from biosearch.config import PlumeConfig, WorldConfig
from biosearch.plume import PuffPlume


def test_puffs_drift_downwind_and_decay() -> None:
    config = PlumeConfig(
        emission_interval=100,
        puffs_per_emission=1,
        intensity_decay=0.1,
        lateral_diffusion=0.0,
        longitudinal_jitter=0.0,
    )
    world = WorldConfig(wind_direction=0.0, wind_speed=0.2, obstacles=())
    plume = PuffPlume(config, world, np.random.default_rng(8))

    plume.step()

    assert len(plume.positions) == 1
    assert plume.positions[0, 0] == pytest.approx(world.source_position[0] + world.wind_speed)
    assert plume.intensities[0] == pytest.approx(0.9)


def test_concentration_is_higher_near_a_puff() -> None:
    config = PlumeConfig(
        emission_interval=100,
        puffs_per_emission=1,
        lateral_diffusion=0.0,
        longitudinal_jitter=0.0,
    )
    world = WorldConfig(obstacles=())
    plume = PuffPlume(config, world, np.random.default_rng(9))
    plume.step()
    puff_position = plume.positions[0].copy()

    readings = plume.concentration_at(
        np.stack((puff_position, puff_position + np.array([5.0, 0.0])))
    )

    assert readings[0] > readings[1]
