import pytest

from tabular.agents import AGENTS
from common.grid_runner import evaluate, train
from common.grid_world import GridWorldEnv
from tabular.type import TrainingConfig


AGENT_CASES = [
    pytest.param(name, 192, 1.0, 0.75, id=name) for name in AGENTS if name != "td-zero"
]


@pytest.mark.parametrize(
    "agent_name, train_episodes, expected_success, expected_return",
    AGENT_CASES,
)
def test_tabular_agent_solves_small_gridworld(
    agent_name: str,
    train_episodes: int,
    expected_success: float,
    expected_return: float,
) -> None:
    config = TrainingConfig()
    env = GridWorldEnv(size=4)
    agent = AGENTS[agent_name](env, config)

    train(env, agent, episodes=train_episodes)
    result = evaluate(env, agent, episodes=16)

    assert result.success_rate() == pytest.approx(expected_success, abs=0.05)
    assert result.mean_return() == pytest.approx(expected_return, abs=0.05)
