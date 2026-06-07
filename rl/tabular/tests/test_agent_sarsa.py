import pytest

from tabular.agents.sarsa import SarsaAgent
from tabular.gridworld import GridWorldEnv
from tabular.type import TrainingConfig


def test_sarsa_updates_pending_transition_on_next_step() -> None:
    config = TrainingConfig(alpha=0.5, gamma=0.8, epsilon_start=0.0)
    env = GridWorldEnv(size=3)
    agent = SarsaAgent(env, config)
    agent.q_values[1] = [0.0, 0.0, 2.0, 0.0]

    action = agent.select_action(0)
    agent.update(
        0,
        action,
        reward=-0.1,
        next_observation=1,
        terminated=False,
        truncated=False,
    )

    assert agent.q_values[0, action] == pytest.approx(0.0)

    next_action = agent.select_action(1)
    agent.update(
        1,
        next_action,
        reward=1.0,
        next_observation=2,
        terminated=True,
        truncated=False,
    )

    assert agent.q_values[0, action] == pytest.approx(0.75)
    assert agent.q_values[1, next_action] == pytest.approx(1.5)
