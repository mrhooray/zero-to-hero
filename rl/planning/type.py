from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    gamma: float = 0.99
    alpha: float = 0.1
    seed: int = 0
