from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# -------------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------------
class Agent(Protocol):
    name: str

    def start_episode(self, episode: int) -> None: ...

    def select_action(self, observation: int, training: bool = True) -> int: ...

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
        truncated: bool,
    ) -> None: ...

    def end_episode(self) -> None: ...


# -------------------------------------------------------------------------
# Config and result
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    gamma: float = 0.99
    alpha: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class EpisodeStats:
    episode: int
    episode_return: float
    length: int
    success: bool


@dataclass(frozen=True)
class TrainingResult:
    agent_name: str
    episodes: list[EpisodeStats]

    def returns(self) -> np.ndarray:
        return np.array(
            [episode.episode_return for episode in self.episodes], dtype=np.float64
        )

    def success_rate(self) -> float:
        return (
            float(np.mean([episode.success for episode in self.episodes]))
            if self.episodes
            else 0.0
        )


@dataclass(frozen=True)
class EvaluationResult:
    agent_name: str
    episodes: list[EpisodeStats]

    def mean_return(self) -> float:
        return (
            float(np.mean([episode.episode_return for episode in self.episodes]))
            if self.episodes
            else 0.0
        )

    def success_rate(self) -> float:
        return (
            float(np.mean([episode.success for episode in self.episodes]))
            if self.episodes
            else 0.0
        )

    def mean_length(self) -> float:
        return sum(episode.length for episode in self.episodes) / max(
            1, len(self.episodes)
        )
