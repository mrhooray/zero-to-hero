import pytest

from tabular.agents.monte_carlo_alpha import MonteCarloAlphaAgent
from tabular.agents.q_learning import QLearningAgent
from tabular.agents.sarsa_expected import SarsaExpectedAgent
from tabular.agents.td_zero import TDZeroAgent
from tabular.gridworld import GridWorldEnv
from tabular.type import TrainingConfig


def test_monte_carlo_bootstraps_final_truncated_step() -> None:
    config = TrainingConfig(alpha=0.5, gamma=0.8)
    env = GridWorldEnv(size=3)
    agent = MonteCarloAlphaAgent(env, config)
    agent.q_values[1] = [1.0, 2.0, 0.0, -1.0]

    agent.update(
        0, 1, reward=-0.1, next_observation=1, terminated=False, truncated=True
    )
    agent.end_episode()

    assert agent.q_values[0, 1] == pytest.approx(0.75)


def test_td_zero_bootstraps_across_truncation() -> None:
    config = TrainingConfig(alpha=0.5, gamma=0.8)
    env = GridWorldEnv(size=3)
    agent = TDZeroAgent(env, config)
    agent.v_values[1] = 2.0

    agent.update(
        0, 1, reward=-0.1, next_observation=1, terminated=False, truncated=True
    )

    assert agent.v_values[0] == pytest.approx(0.75)


def test_td_zero_does_not_bootstrap_across_termination() -> None:
    config = TrainingConfig(alpha=0.5, gamma=0.8)
    env = GridWorldEnv(size=3)
    agent = TDZeroAgent(env, config)
    agent.v_values[1] = 2.0

    agent.update(0, 1, reward=1.0, next_observation=1, terminated=True, truncated=False)

    assert agent.v_values[0] == pytest.approx(0.5)


def test_sarsa_expected_bootstraps_across_truncation() -> None:
    config = TrainingConfig(alpha=0.5, gamma=0.8, epsilon_start=0.0)
    env = GridWorldEnv(size=3)
    agent = SarsaExpectedAgent(env, config)
    agent.q_values[1] = [1.0, 2.0, 0.0, -1.0]

    agent.update(
        0, 1, reward=-0.1, next_observation=1, terminated=False, truncated=True
    )

    assert agent.q_values[0, 1] == pytest.approx(0.75)


def test_q_learning_bootstraps_across_truncation() -> None:
    config = TrainingConfig(alpha=0.5, gamma=0.8)
    env = GridWorldEnv(size=3)
    agent = QLearningAgent(env, config)
    agent.q_values[1] = [1.0, 2.0, 0.0, -1.0]

    agent.update(
        0, 1, reward=-0.1, next_observation=1, terminated=False, truncated=True
    )

    assert agent.q_values[0, 1] == pytest.approx(0.75)
