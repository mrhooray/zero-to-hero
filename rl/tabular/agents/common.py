from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from common.epsilon import epsilon_for_episode
from tabular.type import TrainingConfig


# -------------------------------------------------------------------------
# Core agent
# -------------------------------------------------------------------------
class TabularActionValueAgent:
    name = "tabular-action-value"

    def __init__(self, env: gym.Env[int, int], config: TrainingConfig) -> None:
        self.config = config
        self.observation_count = discrete_size(env.observation_space)
        self.action_count = discrete_size(env.action_space)
        self.q_values = np.zeros(
            (self.observation_count, self.action_count), dtype=np.float64
        )
        self.rng = np.random.default_rng(config.seed)
        self.epsilon = config.epsilon_start

    def start_episode(self, episode: int) -> None:
        self.epsilon = epsilon_for_episode(self.config, episode)

    def select_action(self, observation: int, training: bool = True) -> int:
        if training:
            return epsilon_greedy_action(
                self.q_values[observation], self.epsilon, self.rng
            )
        return greedy_action(self.q_values[observation], self.rng)

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        raise NotImplementedError

    def end_episode(self) -> None:
        pass


class TabularStateValueAgent:
    name = "tabular-state-value"

    def __init__(self, env: gym.Env[int, int], config: TrainingConfig) -> None:
        self.config = config
        self.observation_count = discrete_size(env.observation_space)
        self.action_count = discrete_size(env.action_space)
        self.v_values = np.zeros(self.observation_count, dtype=np.float64)
        self.rng = np.random.default_rng(config.seed)

    def start_episode(self, episode: int) -> None:
        pass

    def select_action(self, observation: int, training: bool = True) -> int:
        return int(self.rng.integers(self.action_count))

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        raise NotImplementedError

    def end_episode(self) -> None:
        pass


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def discrete_size(space: gym.Space) -> int:
    if not isinstance(space, spaces.Discrete):
        raise TypeError("this agent requires Discrete observation and action spaces")
    return int(space.n)


def greedy_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    best_value = np.max(q_values)
    candidates = np.flatnonzero(np.isclose(q_values, best_value))
    return int(rng.choice(candidates))


def epsilon_greedy_action(
    q_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return greedy_action(q_values, rng)


def _mc_end_episode(
    episode: list[tuple[int, int, float, int, bool]],
    first_visit: bool,
    q_values: np.ndarray,
    gamma: float,
    update: Callable[[int, int, float], None],
) -> None:
    returns = [0.0] * len(episode)
    value = 0.0
    last = len(episode) - 1
    for index in range(last, -1, -1):
        _, _, reward, next_observation, truncated = episode[index]
        if truncated and index == last:
            value = reward + gamma * float(np.max(q_values[next_observation]))
        else:
            value = reward + gamma * value
        returns[index] = value

    visited: set[tuple[int, int]] = set()
    for (observation, action, _, _, _), return_value in zip(
        episode, returns, strict=True
    ):
        key = (observation, action)
        if first_visit and key in visited:
            continue
        visited.add(key)
        update(observation, action, return_value)
