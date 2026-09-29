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


@pytest.mark.parametrize(
    ("terminated", "truncated", "expected_delta"),
    [(False, True, 0.9), (True, False, -9.0), (True, True, -9.0)],
)
def test_ppo_advantages_bootstrap_only_at_truncation(
    terminated: bool, truncated: bool, expected_delta: float
) -> None:
    config = TrainingConfig(gamma=0.99, hidden_size=16)
    agent = PPOAgent(gym.make("CartPole-v1"), config, PPOConfig(gae_lambda=0.95))
    agent.value = torch.nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        agent.value.weight.fill_(2.5)
    agent.rollout = [
        _rollout_step(reward=1.0, terminated=False),
        _rollout_step(reward=1.0, terminated=terminated, truncated=truncated),
    ]
    agent.rollout[-1].transition.next_observation[:] = 1.0

    advantages = agent._advantages(torch.as_tensor([10.0, 10.0]))

    assert advantages.tolist() == pytest.approx(
        [
            0.9 + config.gamma * agent.ppo_config.gae_lambda * expected_delta,
            expected_delta,
        ]
    )


def _rollout_step(
    reward: float, terminated: bool, truncated: bool = False
) -> RolloutStep:
    return RolloutStep(
        transition=Transition(
            observation=np.zeros(4, dtype=np.float32),
            action=0,
            reward=reward,
            next_observation=np.zeros(4, dtype=np.float32),
            terminated=terminated,
            truncated=truncated,
        ),
        log_prob=torch.as_tensor(0.0),
        value=torch.as_tensor(0.0),
    )
