from __future__ import annotations

from typing import Protocol


class _HasEpsilonSchedule(Protocol):
    @property
    def epsilon_start(self) -> float: ...
    @property
    def epsilon_end(self) -> float: ...
    @property
    def epsilon_decay_episodes(self) -> int: ...


def epsilon_for_episode(config: _HasEpsilonSchedule, episode: int) -> float:
    decay_episodes = max(1, config.epsilon_decay_episodes)
    progress = min(episode / decay_episodes, 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
