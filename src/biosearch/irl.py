"""Small AIRL experiment built around synthetic moth-controller demonstrations.

The implementation keeps the research boundary explicit. Demonstrations and
learned policies receive the same 13 local observations. Source position and
distance are never inputs to behavior cloning, the discriminator, or the
learned reward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi
from statistics import fmean
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from biosearch.ablations import exact_mcnemar_p_value
from biosearch.config import SimulationConfig
from biosearch.controllers import MothController
from biosearch.evaluation import sample_start_pose, success_rate_interval
from biosearch.gym_environment import BioSearchEnv

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class TransitionBatch:
    """Flat transitions from one or more complete episodes."""

    observations: FloatArray
    actions: IntArray
    next_observations: FloatArray
    dones: BoolArray

    def __len__(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class DemonstrationEpisode:
    """Outcome metadata for one attempted demonstration episode."""

    seed: int
    success: bool
    steps: int
    retained: bool


@dataclass(frozen=True)
class PolicyEpisode:
    """Evaluation result for one learned policy episode."""

    policy: str
    seed: int
    success: bool
    steps: int
    collisions: int


@dataclass(frozen=True)
class DiscriminatorMetrics:
    """Mean discriminator loss and classification accuracy."""

    loss: float
    accuracy: float


def experiment_config(max_steps: int = 600) -> SimulationConfig:
    """Return the fixed world used for demonstration learning."""

    base = SimulationConfig()
    return replace(base, agent=replace(base.agent, max_steps=max_steps))


def _empty_transition_lists() -> tuple[list[FloatArray], list[int], list[FloatArray], list[bool]]:
    return [], [], [], []


def _stack_transitions(
    observations: list[FloatArray],
    actions: list[int],
    next_observations: list[FloatArray],
    dones: list[bool],
) -> TransitionBatch:
    if not observations:
        raise ValueError("At least one transition is required.")
    return TransitionBatch(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
    )


def collect_moth_demonstrations(
    *,
    successful_episodes: int,
    seed_start: int,
    max_attempts: int = 500,
    max_steps: int = 600,
) -> tuple[TransitionBatch, list[DemonstrationEpisode]]:
    """Collect the first requested successful moth-controller episodes.

    Failed attempts remain in the returned metadata but are not treated as
    expert transitions. This prevents time-limit behavior from being labeled
    as a successful demonstration.
    """

    if successful_episodes <= 0 or max_attempts <= 0:
        raise ValueError("Demonstration counts must be positive.")
    config = experiment_config(max_steps)
    environment = BioSearchEnv(config, randomize_start=False)
    kept = 0
    all_observations: list[FloatArray] = []
    all_actions: list[int] = []
    all_next_observations: list[FloatArray] = []
    all_dones: list[bool] = []
    episodes: list[DemonstrationEpisode] = []
    try:
        for seed in range(seed_start, seed_start + max_attempts):
            position, heading = sample_start_pose(seed, config)
            observation, _ = environment.reset(
                seed=seed,
                options={
                    "agent_position": position,
                    "heading": heading,
                    "simulation_seed": seed,
                },
            )
            controller = MothController()
            episode_observations, episode_actions, episode_next, episode_dones = (
                _empty_transition_lists()
            )
            done = False
            while not done:
                relative_heading = float(
                    (environment.simulation.heading - config.world.wind_direction + pi) % (2 * pi)
                    - pi
                )
                action = int(
                    controller.act(
                        environment.simulation.last_sensor_reading,
                        heading_relative_to_wind=relative_heading,
                    )
                )
                next_observation, _, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                episode_observations.append(observation.copy())
                episode_actions.append(action)
                episode_next.append(next_observation.copy())
                episode_dones.append(done)
                observation = next_observation

            success = environment.simulation.success
            retained = bool(success and kept < successful_episodes)
            episodes.append(
                DemonstrationEpisode(
                    seed=seed,
                    success=success,
                    steps=environment.simulation.step_count,
                    retained=retained,
                )
            )
            if retained:
                all_observations.extend(episode_observations)
                all_actions.extend(episode_actions)
                all_next_observations.extend(episode_next)
                all_dones.extend(episode_dones)
                kept += 1
            if kept == successful_episodes:
                break
    finally:
        environment.close()
    if kept != successful_episodes:
        raise RuntimeError(
            f"Collected {kept} successful demonstrations after {max_attempts} attempts."
        )
    return (
        _stack_transitions(
            all_observations,
            all_actions,
            all_next_observations,
            all_dones,
        ),
        episodes,
    )


def make_ppo(
    *,
    seed: int,
    n_envs: int = 1,
    max_steps: int = 600,
    verbose: int = 0,
) -> PPO:
    """Construct the policy architecture used by BC and AIRL."""

    if n_envs <= 0:
        raise ValueError("n_envs must be positive.")
    config = experiment_config(max_steps)
    vector_environment = DummyVecEnv(
        [lambda config=config: BioSearchEnv(config, randomize_start=True) for _ in range(n_envs)]
    )
    return PPO(
        "MlpPolicy",
        vector_environment,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        device="cpu",
        verbose=verbose,
    )


def behavior_clone(
    model: PPO,
    demonstrations: TransitionBatch,
    *,
    epochs: int = 30,
    batch_size: int = 1_024,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Fit the PPO actor to expert actions with capped class balancing."""

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("Behavior-cloning settings must be positive.")
    rng = np.random.default_rng(seed)
    device = model.policy.device
    observations = demonstrations.observations
    actions = demonstrations.actions
    counts = np.bincount(actions, minlength=model.action_space.n)
    weights = len(actions) / (model.action_space.n * np.maximum(counts, 1))
    weights = np.minimum(weights, 10.0)
    class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(len(actions))
        epoch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            observation_tensor = torch.as_tensor(
                observations[indices], dtype=torch.float32, device=device
            )
            action_tensor = torch.as_tensor(actions[indices], dtype=torch.long, device=device)
            distribution = model.policy.get_distribution(observation_tensor)
            logits = distribution.distribution.logits
            loss = nn.functional.cross_entropy(logits, action_tensor, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(fmean(epoch_losses))
    return losses


def action_accuracy(model: PPO, transitions: TransitionBatch) -> float:
    """Return deterministic action agreement on a transition batch."""

    predictions, _ = model.predict(transitions.observations, deterministic=True)
    return float(np.mean(np.asarray(predictions, dtype=np.int64) == transitions.actions))


class AIRLRewardNetwork(nn.Module):
    """State-action reward plus potential shaping used by AIRL."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        *,
        hidden_size: int = 128,
        gamma: float = 0.995,
    ) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.action_count = action_count
        self.gamma = gamma
        self.reward = nn.Sequential(
            nn.Linear(observation_size + action_count, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.potential = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def state_action_reward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return the unshaped reward component g(s, a)."""

        one_hot = nn.functional.one_hot(actions.long(), self.action_count).float()
        inputs = torch.cat((observations.float(), one_hot), dim=1)
        return self.reward(inputs).squeeze(1)

    def shaped_reward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Return g(s,a) + gamma*h(s') - h(s)."""

        base = self.state_action_reward(observations, actions)
        current_potential = self.potential(observations.float()).squeeze(1)
        next_potential = self.potential(next_observations.float()).squeeze(1)
        continuation = 1.0 - dones.float()
        return base + self.gamma * continuation * next_potential - current_potential

    def discriminator_logits(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        log_policy_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Return log D - log(1-D) for the AIRL discriminator."""

        return self.shaped_reward(observations, actions, next_observations, dones) - (
            log_policy_probability.detach()
        )


def policy_log_probabilities(model: PPO, transitions: TransitionBatch) -> torch.Tensor:
    """Evaluate transition actions under the current generator policy."""

    device = model.policy.device
    observations = torch.as_tensor(transitions.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(transitions.actions, dtype=torch.long, device=device)
    with torch.no_grad():
        _, log_probabilities, _ = model.policy.evaluate_actions(observations, actions)
    return log_probabilities.detach().cpu()


def collect_policy_transitions(
    model: PPO,
    *,
    minimum_transitions: int,
    seed_start: int,
    max_steps: int = 600,
    deterministic: bool = False,
) -> TransitionBatch:
    """Collect complete generator episodes until a transition target is met."""

    if minimum_transitions <= 0:
        raise ValueError("minimum_transitions must be positive.")
    config = experiment_config(max_steps)
    environment = BioSearchEnv(config, randomize_start=False)
    observations, actions, next_observations, dones = _empty_transition_lists()
    seed = seed_start
    try:
        while len(actions) < minimum_transitions:
            position, heading = sample_start_pose(seed, config)
            observation, _ = environment.reset(
                seed=seed,
                options={
                    "agent_position": position,
                    "heading": heading,
                    "simulation_seed": seed,
                },
            )
            done = False
            while not done:
                predicted, _ = model.predict(observation, deterministic=deterministic)
                action = int(np.asarray(predicted).item())
                next_observation, _, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                observations.append(observation.copy())
                actions.append(action)
                next_observations.append(next_observation.copy())
                dones.append(done)
                observation = next_observation
            seed += 1
    finally:
        environment.close()
    return _stack_transitions(observations, actions, next_observations, dones)


def _transition_tensors(
    transitions: TransitionBatch,
    indices: NDArray[np.int64],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(transitions.observations[indices], dtype=torch.float32, device=device),
        torch.as_tensor(transitions.actions[indices], dtype=torch.long, device=device),
        torch.as_tensor(transitions.next_observations[indices], dtype=torch.float32, device=device),
        torch.as_tensor(transitions.dones[indices], dtype=torch.float32, device=device),
    )


def train_airl_discriminator(
    reward_network: AIRLRewardNetwork,
    generator: PPO,
    expert: TransitionBatch,
    generated: TransitionBatch,
    *,
    epochs: int = 8,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> DiscriminatorMetrics:
    """Train one balanced discriminator round."""

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("Discriminator settings must be positive.")
    device = next(reward_network.parameters()).device
    expert_log_probs = policy_log_probabilities(generator, expert)
    generated_log_probs = policy_log_probabilities(generator, generated)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(reward_network.parameters(), lr=learning_rate)
    losses: list[float] = []
    accuracies: list[float] = []
    steps_per_epoch = max(len(expert), len(generated)) // batch_size + 1
    reward_network.train()
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            expert_indices = rng.integers(0, len(expert), size=batch_size)
            generated_indices = rng.integers(0, len(generated), size=batch_size)
            expert_tensors = _transition_tensors(expert, expert_indices, device=device)
            generated_tensors = _transition_tensors(generated, generated_indices, device=device)
            expert_logits = reward_network.discriminator_logits(
                *expert_tensors,
                expert_log_probs[expert_indices].to(device),
            )
            generated_logits = reward_network.discriminator_logits(
                *generated_tensors,
                generated_log_probs[generated_indices].to(device),
            )
            logits = torch.cat((expert_logits, generated_logits))
            labels = torch.cat(
                (
                    torch.ones(batch_size, device=device),
                    torch.zeros(batch_size, device=device),
                )
            )
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            regularization = 1e-5 * sum(
                parameter.square().sum() for parameter in reward_network.parameters()
            )
            total_loss = loss + regularization
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(reward_network.parameters(), max_norm=5.0)
            optimizer.step()
            with torch.no_grad():
                accuracy = ((logits >= 0) == labels.bool()).float().mean()
            losses.append(float(loss.detach().cpu()))
            accuracies.append(float(accuracy.detach().cpu()))
    reward_network.eval()
    return DiscriminatorMetrics(loss=fmean(losses), accuracy=fmean(accuracies))


class AIRLRewardWrapper(gym.Wrapper):
    """Replace privileged environment reward with the AIRL discriminator reward."""

    def __init__(
        self,
        environment: BioSearchEnv,
        reward_network: AIRLRewardNetwork,
        generator: PPO,
    ) -> None:
        super().__init__(environment)
        self.reward_network = reward_network
        self.generator = generator
        self._last_observation: FloatArray | None = None

    def reset(self, **kwargs: Any) -> tuple[FloatArray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._last_observation = observation.copy()
        return observation, info

    def step(self, action: int) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        if self._last_observation is None:
            raise RuntimeError("reset() must be called before step().")
        next_observation, _ignored_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        device = self.generator.policy.device
        observation_tensor = torch.as_tensor(
            self._last_observation[None, :], dtype=torch.float32, device=device
        )
        action_tensor = torch.as_tensor([action], dtype=torch.long, device=device)
        next_tensor = torch.as_tensor(next_observation[None, :], dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor([done], dtype=torch.float32, device=device)
        with torch.no_grad():
            _, log_probability, _ = self.generator.policy.evaluate_actions(
                observation_tensor, action_tensor
            )
            learned_reward = self.reward_network.discriminator_logits(
                observation_tensor,
                action_tensor,
                next_tensor,
                done_tensor,
                log_probability,
            )[0]
        self._last_observation = next_observation.copy()
        return (
            next_observation,
            float(torch.clamp(learned_reward, -20.0, 20.0)),
            terminated,
            truncated,
            info,
        )


def attach_airl_reward_environment(
    generator: PPO,
    reward_network: AIRLRewardNetwork,
    *,
    n_envs: int,
    seed: int,
    max_steps: int = 600,
) -> None:
    """Replace a generator's environment with shared AIRL reward wrappers."""

    config = experiment_config(max_steps)

    def make_environment() -> AIRLRewardWrapper:
        base = BioSearchEnv(config, randomize_start=True)
        return AIRLRewardWrapper(base, reward_network, generator)

    vector_environment = DummyVecEnv([make_environment for _ in range(n_envs)])
    vector_environment.seed(seed)
    generator.set_env(vector_environment)


def evaluate_policy(
    model: PPO,
    *,
    policy_name: str,
    seeds: range,
    max_steps: int = 600,
) -> list[PolicyEpisode]:
    """Evaluate a learned policy with the physical success criterion."""

    config = experiment_config(max_steps)
    environment = BioSearchEnv(config, randomize_start=False)
    rows: list[PolicyEpisode] = []
    try:
        for seed in seeds:
            position, heading = sample_start_pose(seed, config)
            observation, _ = environment.reset(
                seed=seed,
                options={
                    "agent_position": position,
                    "heading": heading,
                    "simulation_seed": seed,
                },
            )
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = environment.step(
                    int(np.asarray(action).item())
                )
                done = terminated or truncated
            rows.append(
                PolicyEpisode(
                    policy=policy_name,
                    seed=seed,
                    success=environment.simulation.success,
                    steps=environment.simulation.step_count,
                    collisions=environment.simulation.collision_count,
                )
            )
    finally:
        environment.close()
    return rows


def summarize_policy(rows: list[PolicyEpisode]) -> dict[str, float | int | str]:
    """Summarize success and search time for one policy."""

    if not rows:
        raise ValueError("At least one policy episode is required.")
    successes = sum(row.success for row in rows)
    ci_low, ci_high = success_rate_interval(successes, len(rows))
    successful_steps = [row.steps for row in rows if row.success]
    return {
        "policy": rows[0].policy,
        "episodes": len(rows),
        "success_rate": 100.0 * successes / len(rows),
        "success_ci95_low": ci_low,
        "success_ci95_high": ci_high,
        "mean_success_steps": fmean(successful_steps) if successful_steps else float("nan"),
        "mean_collisions": fmean(row.collisions for row in rows),
    }


def compare_paired_success(
    first: list[PolicyEpisode],
    second: list[PolicyEpisode],
) -> dict[str, float | int | str]:
    """Compare two policies evaluated on identical seeds."""

    first_by_seed = {row.seed: row.success for row in first}
    second_by_seed = {row.seed: row.success for row in second}
    if not first_by_seed or set(first_by_seed) != set(second_by_seed):
        raise ValueError("Paired policy rows must contain identical non-empty seed sets.")
    both = sum(first_by_seed[seed] and second_by_seed[seed] for seed in first_by_seed)
    first_only = sum(first_by_seed[seed] and not second_by_seed[seed] for seed in first_by_seed)
    second_only = sum(not first_by_seed[seed] and second_by_seed[seed] for seed in first_by_seed)
    neither = len(first_by_seed) - both - first_only - second_only
    return {
        "first_policy": first[0].policy,
        "second_policy": second[0].policy,
        "episodes": len(first_by_seed),
        "both_succeed": both,
        "first_only_succeeds": first_only,
        "second_only_succeeds": second_only,
        "neither_succeeds": neither,
        "second_minus_first_percentage_points": (
            100.0 * (second_only - first_only) / len(first_by_seed)
        ),
        "exact_mcnemar_p_value": exact_mcnemar_p_value(first_only, second_only),
    }
