from __future__ import annotations

from tabular.agents.common import TabularStateValueAgent


class TDZeroAgent(TabularStateValueAgent):
    name = "td-zero"

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
            target += self.config.gamma * self.v_values[next_observation]
        self.v_values[observation] += self.config.alpha * (
            target - self.v_values[observation]
        )
