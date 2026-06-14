from __future__ import annotations

import gymnasium as gym
import numpy as np

from common.type import EpisodeStats, RunResult
from deep.type import Agent


# -------------------------------------------------------------------------
# Public runners
# -------------------------------------------------------------------------
def train(
    env: gym.Env,
    agent: Agent,
    episodes: int,
    seed: int,
) -> RunResult:
    stats = []
    for episode in range(episodes):
        stats.append(_run_episode(env, agent, episode, seed=seed, training=True))

    return RunResult(agent.name, stats)


def evaluate(
    env: gym.Env,
    agent: Agent,
    episodes: int,
    seed: int,
) -> RunResult:
    stats = []
    for episode in range(episodes):
        stats.append(_run_episode(env, agent, episode, seed=seed, training=False))

    return RunResult(agent.name, stats)


# -------------------------------------------------------------------------
# Private utilities
# -------------------------------------------------------------------------
def _run_episode(
    env: gym.Env,
    agent: Agent,
    episode: int,
    seed: int,
    training: bool,
) -> EpisodeStats:
    observation, _ = env.reset(seed=seed + episode)
    env.action_space.seed(seed + episode)
    observation = np.asarray(observation, dtype=np.float32)
    agent.start_episode(episode)
    episode_return = 0.0
    length = 0

    while True:
        action = agent.select_action(observation, training=training)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)
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
        length += 1
        observation = next_observation
        if terminated or truncated:
            break

    agent.end_episode()
    return EpisodeStats(
        episode,
        episode_return,
        length,
        terminated,
        truncated,
        is_success=truncated,
    )
