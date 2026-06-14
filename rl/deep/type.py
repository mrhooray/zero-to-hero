from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


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
