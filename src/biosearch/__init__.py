"""BioSearch RL: a small, reproducible odor-source search simulation."""

from gymnasium.envs.registration import register, registry

from biosearch.config import SimulationConfig
from biosearch.environment import Action, BioSearchSimulation, StepResult
from biosearch.gym_environment import BioSearchEnv, RewardConfig

if "BioSearch-v0" not in registry:
    register(
        id="BioSearch-v0",
        entry_point="biosearch.gym_environment:BioSearchEnv",
    )

__all__ = [
    "Action",
    "BioSearchEnv",
    "BioSearchSimulation",
    "RewardConfig",
    "SimulationConfig",
    "StepResult",
]
__version__ = "0.3.0"
