from __future__ import annotations

import gymnasium as gym
import numpy as np

from tabular.agents.common import TabularActionValueAgent
from tabular.type import TrainingConfig


class DynaQAgent(TabularActionValueAgent):
    name = "dyna-q"

    def __init__(
        self,
        env: gym.Env[int, int],
        config: TrainingConfig,
        planning_steps: int = 8,
    ) -> None:
        super().__init__(env, config)
        self.planning_steps = planning_steps
        self.model: dict[tuple[int, int], tuple[int, float, bool]] = {}
        self.visited: set[tuple[int, int]] = set()
        self.visited_keys: list[tuple[int, int]] = []

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._q_update(observation, action, reward, next_observation, terminated)

        key = (observation, action)
        self.model[key] = (next_observation, reward, terminated)
        if key not in self.visited:
            self.visited_keys.append(key)
            self.visited.add(key)

        for _ in range(self.planning_steps):
            sim_obs, sim_act = self._visited_choices()
            sim_next, sim_reward, sim_term = self.model[(sim_obs, sim_act)]
            self._q_update(sim_obs, sim_act, sim_reward, sim_next, sim_term)

    # -------------------------------------------------------------------------
    # Q-learning
    # -------------------------------------------------------------------------

    def _q_update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
    ) -> None:
        target = reward
        if not terminated:
            target += self.config.gamma * float(np.max(self.q_values[next_observation]))
        self.q_values[observation, action] += self.config.alpha * (
            target - self.q_values[observation, action]
        )

    # -------------------------------------------------------------------------
    # Model sampling
    # -------------------------------------------------------------------------

    def _visited_choices(self) -> tuple[int, int]:
        return self.rng.choice(self.visited_keys)
