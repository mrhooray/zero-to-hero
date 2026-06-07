from __future__ import annotations

import gymnasium as gym

from tabular.agents.common import TabularActionValueAgent
from tabular.type import TrainingConfig


class SarsaAgent(TabularActionValueAgent):
    name = "sarsa"

    def __init__(self, env: gym.Env[int, int], config: TrainingConfig) -> None:
        super().__init__(env, config)
        self.pending_transition: tuple[int, int, float, int] | None = None

    def start_episode(self, episode: int) -> None:
        super().start_episode(episode)
        self.pending_transition = None

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        # Finish the previous transition with this action as A_{t+1}
        if self.pending_transition is not None:
            self._apply_update(*self.pending_transition, next_action=action)

        # The episode ends here, so it has no next action.
        if terminated or truncated:
            self._apply_update(
                observation,
                action,
                reward,
                next_observation,
                next_action=None,
            )
            self.pending_transition = None
            return

        self.pending_transition = (observation, action, reward, next_observation)

    def end_episode(self) -> None:
        self.pending_transition = None

    def _apply_update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        next_action: int | None,
    ) -> None:
        target = reward
        if next_action is not None:
            target += self.config.gamma * self.q_values[next_observation, next_action]
        self.q_values[observation, action] += self.config.alpha * (
            target - self.q_values[observation, action]
        )
