from __future__ import annotations

import numpy as np

from tabular.agents.common import TabularActionValueAgent


class SarsaExpectedAgent(TabularActionValueAgent):
    name = "sarsa-expected"

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
            probs = _epsilon_greedy_probs(
                self.q_values[next_observation],
                self.epsilon,
            )
            target += self.config.gamma * float(
                np.dot(probs, self.q_values[next_observation])
            )
        self.q_values[observation, action] += self.config.alpha * (
            target - self.q_values[observation, action]
        )


def _epsilon_greedy_probs(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    action_count = len(q_values)
    probs = np.full(action_count, epsilon / action_count, dtype=np.float64)
    best_value = np.max(q_values)
    candidates = np.flatnonzero(np.isclose(q_values, best_value))
    probs[candidates] += (1.0 - epsilon) / len(candidates)
    return probs
