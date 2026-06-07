from __future__ import annotations

import numpy as np

from tabular.agents.common import TabularActionValueAgent


class QLearningAgent(TabularActionValueAgent):
    name = "q-learning"

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        target = reward
        if not terminated:
            target += self.config.gamma * float(np.max(self.q_values[next_observation]))
        self.q_values[observation, action] += self.config.alpha * (
            target - self.q_values[observation, action]
        )
