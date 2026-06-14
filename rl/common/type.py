from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeStats:
    episode: int
    episode_return: float
    length: int
    terminated: bool
    truncated: bool
    is_success: bool


@dataclass(frozen=True)
class RunResult:
    agent_name: str
    episodes: list[EpisodeStats]

    def returns(self) -> np.ndarray:
        return np.array(
            [episode.episode_return for episode in self.episodes], dtype=np.float64
        )

    def mean_return(self) -> float:
        return (
            float(np.mean([episode.episode_return for episode in self.episodes]))
            if self.episodes
            else 0.0
        )

    def mean_length(self) -> float:
        return sum(episode.length for episode in self.episodes) / max(
            1, len(self.episodes)
        )

    def termination_rate(self) -> float:
        return (
            float(np.mean([episode.terminated for episode in self.episodes]))
            if self.episodes
            else 0.0
        )

    def truncation_rate(self) -> float:
        return (
            float(np.mean([episode.truncated for episode in self.episodes]))
            if self.episodes
            else 0.0
        )

    def success_rate(self) -> float:
        return (
            float(np.mean([episode.is_success for episode in self.episodes]))
            if self.episodes
            else 0.0
        )
