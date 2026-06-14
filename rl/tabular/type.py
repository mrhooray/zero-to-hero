from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    gamma: float = 0.99
    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.04
    epsilon_decay_episodes: int = 200
    seed: int = 0
