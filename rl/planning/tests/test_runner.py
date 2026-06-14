import pytest

from planning.agents import AGENTS
from common.grid_runner import evaluate
from common.grid_world import GridWorldEnv
from planning.type import TrainingConfig


@pytest.mark.parametrize("agent_name", AGENTS.keys())
def test_planning_agent_solves_small_gridworld(agent_name: str) -> None:
    env = GridWorldEnv(size=4)
    agent = AGENTS[agent_name](env, TrainingConfig(seed=24))

    result = evaluate(env, agent, episodes=16)

    assert result.success_rate() == pytest.approx(1.0)
    assert result.mean_return() == pytest.approx(0.75)
    assert result.mean_length() == pytest.approx(6.0)
