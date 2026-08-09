from __future__ import annotations

import numpy as np
import torch

from biosearch.irl import (
    AIRLRewardNetwork,
    PolicyEpisode,
    TransitionBatch,
    collect_moth_demonstrations,
    compare_paired_success,
)


def test_airl_reward_shapes_match_transition_batch() -> None:
    network = AIRLRewardNetwork(observation_size=13, action_count=6, hidden_size=16)
    observations = torch.zeros((5, 13))
    actions = torch.tensor([0, 1, 2, 3, 0])
    next_observations = torch.ones((5, 13))
    dones = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)
    log_probabilities = torch.full((5,), -1.0)

    reward = network.state_action_reward(observations, actions)
    shaped = network.shaped_reward(observations, actions, next_observations, dones)
    logits = network.discriminator_logits(
        observations,
        actions,
        next_observations,
        dones,
        log_probabilities,
    )

    assert reward.shape == (5,)
    assert shaped.shape == (5,)
    assert logits.shape == (5,)
    torch.testing.assert_close(logits, shaped + 1.0)


def test_transition_batch_length_uses_actions() -> None:
    batch = TransitionBatch(
        observations=np.zeros((3, 13), dtype=np.float32),
        actions=np.zeros(3, dtype=np.int64),
        next_observations=np.zeros((3, 13), dtype=np.float32),
        dones=np.zeros(3, dtype=np.bool_),
    )

    assert len(batch) == 3


def test_demonstrations_are_seeded_and_keep_only_successes() -> None:
    first, first_episodes = collect_moth_demonstrations(
        successful_episodes=2,
        seed_start=70,
        max_attempts=20,
        max_steps=600,
    )
    second, second_episodes = collect_moth_demonstrations(
        successful_episodes=2,
        seed_start=70,
        max_attempts=20,
        max_steps=600,
    )

    np.testing.assert_array_equal(first.observations, second.observations)
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first_episodes == second_episodes
    assert sum(episode.retained for episode in first_episodes) == 2
    assert all(episode.success for episode in first_episodes if episode.retained)


def test_paired_success_comparison_counts_discordant_seeds() -> None:
    first = [
        PolicyEpisode("bc", 1, True, 10, 0),
        PolicyEpisode("bc", 2, True, 10, 0),
        PolicyEpisode("bc", 3, False, 10, 0),
    ]
    second = [
        PolicyEpisode("airl", 1, True, 10, 0),
        PolicyEpisode("airl", 2, False, 10, 0),
        PolicyEpisode("airl", 3, True, 10, 0),
    ]

    comparison = compare_paired_success(first, second)

    assert comparison["both_succeed"] == 1
    assert comparison["first_only_succeeds"] == 1
    assert comparison["second_only_succeeds"] == 1
    assert comparison["neither_succeeds"] == 0
    assert comparison["second_minus_first_percentage_points"] == 0.0
    assert comparison["exact_mcnemar_p_value"] == 1.0
