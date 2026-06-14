from __future__ import annotations

import gymnasium as gym

from common.grid_type import Agent
from common.type import EpisodeStats, RunResult


# -------------------------------------------------------------------------
# Public runners
# -------------------------------------------------------------------------
def train(
    env: gym.Env[int, int],
    agent: Agent,
    episodes: int,
) -> RunResult:
    stats = []
    for episode in range(episodes):
        stats.append(_run_episode(env, agent, episode=episode, training=True))

    return RunResult(agent.name, stats)


def evaluate(
    env: gym.Env[int, int],
    agent: Agent,
    episodes: int,
) -> RunResult:
    stats = []
    for episode in range(episodes):
        stats.append(
            _run_episode(
                env,
                agent,
                episode=episode,
                training=False,
            )
        )

    return RunResult(agent.name, stats)


# -------------------------------------------------------------------------
# Private utilities
# -------------------------------------------------------------------------
def _run_episode(
    env: gym.Env[int, int],
    agent: Agent,
    episode: int,
    training: bool,
) -> EpisodeStats:
    observation, _ = env.reset()
    agent.start_episode(episode)
    episode_return = 0.0
    is_success = False

    while True:
        action = agent.select_action(observation, training=training)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if training:
            agent.update(
                observation,
                action,
                float(reward),
                next_observation,
                terminated,
                truncated,
            )

        episode_return += float(reward)
        is_success = is_success or bool(info.get("is_success", False))
        observation = next_observation
        if terminated or truncated:
            break

    agent.end_episode()
    return EpisodeStats(
        episode,
        episode_return,
        info["steps"],
        terminated,
        truncated,
        is_success,
    )
