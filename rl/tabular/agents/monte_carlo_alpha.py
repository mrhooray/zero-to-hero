from __future__ import annotations

import gymnasium as gym

from tabular.agents.common import TabularActionValueAgent, _mc_end_episode
from tabular.type import TrainingConfig


class MonteCarloAlphaAgent(TabularActionValueAgent):
    name = "mc-alpha"

    def __init__(
        self,
        env: gym.Env[int, int],
        config: TrainingConfig,
        first_visit: bool = True,
    ) -> None:
        super().__init__(env, config)
        self.first_visit = first_visit
        self.episode: list[tuple[int, int, float, int, bool]] = []

    def start_episode(self, episode: int) -> None:
        super().start_episode(episode)
        self.episode = []

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.episode.append((observation, action, reward, next_observation, truncated))

    def end_episode(self) -> None:
        alpha = self.config.alpha

        def _update(obs: int, act: int, ret: float) -> None:
            self.q_values[obs, act] += alpha * (ret - self.q_values[obs, act])

        _mc_end_episode(
            self.episode,
            self.first_visit,
            self.q_values,
            self.config.gamma,
            _update,
        )
