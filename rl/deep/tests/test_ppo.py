import gymnasium as gym
import numpy as np
import pytest
import torch

from deep.agents.common import Transition
from deep.agents.ppo import PPOAgent, PPOConfig, RolloutStep
from deep.type import TrainingConfig


def test_ppo_stores_rollout_step_after_update() -> None:
    config = TrainingConfig(hidden_size=16, seed=24)
    agent = PPOAgent(gym.make("CartPole-v1"), config)
    observation = np.zeros(4, dtype=np.float32)
    next_observation = np.ones(4, dtype=np.float32)

    action = agent.select_action(observation)
    agent.update(
        observation,
        action,
        reward=1.0,
        next_observation=next_observation,
        terminated=False,
        truncated=True,
    )

    assert agent.pending_step is None
    assert len(agent.rollout) == 1
    assert agent.rollout[0].transition.truncated
    assert agent.rollout[0].log_prob.shape == torch.Size([])
    assert agent.rollout[0].value.shape == torch.Size([])


def test_ppo_advantages_use_next_step_value() -> None:
    config = TrainingConfig(gamma=0.8, hidden_size=16)
    agent = PPOAgent(gym.make("CartPole-v1"), config, PPOConfig(gae_lambda=0.5))
    agent.rollout = [
        _rollout_step(reward=1.0, terminated=False),
        _rollout_step(reward=3.0, terminated=True),
    ]

    advantages = agent._advantages(torch.as_tensor([0.5, 1.0]))

    assert advantages.tolist() == pytest.approx([2.1, 2.0])


def _rollout_step(reward: float, terminated: bool) -> RolloutStep:
    return RolloutStep(
        transition=Transition(
            observation=np.zeros(4, dtype=np.float32),
            action=0,
            reward=reward,
            next_observation=np.zeros(4, dtype=np.float32),
            terminated=terminated,
            truncated=False,
        ),
        log_prob=torch.as_tensor(0.0),
        value=torch.as_tensor(0.0),
    )
