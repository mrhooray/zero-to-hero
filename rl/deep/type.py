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

    def select_action(self, observation: np.ndarray, training: bool = True) -> int: ...

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None: ...

    def end_episode(self) -> None: ...


# -------------------------------------------------------------------------
# Config and result
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    # Shared
    gamma: float = 0.99
    lr: float = 3e-4
    seed: int = 24
    hidden_size: int = 128
    debug: bool = False

    # Replay-based agents
    batch_size: int = 128
    replay_size: int = 1024 * 16
    warmup_steps: int = 1024 * 2

    # DQN
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.05
    dqn_epsilon_decay_episodes: int = 256
    dqn_target_tau: float = 0.005
    dqn_gradient_clip: float = 1

    # SAC
    sac_alpha: float = 0.2
    sac_target_tau: float = 0.005


@dataclass(frozen=True)
class EpisodeStats:
    episode: int
    episode_return: float
    length: int
    terminated: bool
    truncated: bool


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

    def success_rate(self) -> float:
        return (
            float(np.mean([episode.truncated for episode in self.episodes]))
            if self.episodes
            else 0.0
        )
