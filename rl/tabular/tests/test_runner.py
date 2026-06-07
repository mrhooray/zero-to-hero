from tabular.agents.q_learning import QLearningAgent
from tabular.gridworld import GridWorldEnv
from tabular.runner import evaluate, train
from tabular.type import TrainingConfig


def test_q_learning_solves_small_gridworld() -> None:
    config = TrainingConfig(
        alpha=0.4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=128,
        seed=24,
    )
    env = GridWorldEnv(size=4)
    agent = QLearningAgent(env, config)

    train(env, agent, episodes=256)
    result = evaluate(env, agent, episodes=32)

    assert result.success_rate() >= 0.8
    assert result.mean_return() > 0.4
