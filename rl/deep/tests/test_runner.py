import gymnasium as gym
import pytest

from deep.agents import AGENTS
from deep.runner import evaluate, train
from deep.type import TrainingConfig


@pytest.mark.parametrize("agent_name", AGENTS.keys())
def test_runner_trains_and_evaluates_deep_agents(agent_name: str) -> None:
    config = TrainingConfig(
        batch_size=4,
        warmup_steps=4,
        hidden_size=16,
        dqn_epsilon_decay_episodes=1,
        seed=24,
    )
    env = gym.make("CartPole-v1")
    agent = AGENTS[agent_name](env, config)

    train_result = train(env, agent, episodes=16, seed=config.seed)
    eval_result = evaluate(
        gym.make("CartPole-v1"),
        agent,
        episodes=4,
        seed=config.seed,
    )

    assert train_result.episodes[0].terminated or train_result.episodes[0].truncated
    assert eval_result.mean_return() >= 8
    assert eval_result.termination_rate() + eval_result.success_rate() == 1.0
