from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn


# -------------------------------------------------------------------------
# Models and replay
# -------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.transitions: deque[Transition] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.transitions)

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = self.rng.choice(len(self.transitions), size=batch_size, replace=False)
        batch = [self.transitions[int(index)] for index in indices]
        observations = torch.as_tensor(
            np.array([item.observation for item in batch]), dtype=torch.float32
        )
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32)
        next_observations = torch.as_tensor(
            np.array([item.next_observation for item in batch]), dtype=torch.float32
        )
        terminated = torch.as_tensor(
            [item.terminated for item in batch], dtype=torch.float32
        )
        return observations, actions, rewards, next_observations, terminated


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def cartpole_sizes(env: gym.Env) -> tuple[int, int]:
    if not isinstance(env.observation_space, spaces.Box):
        raise TypeError("CartPole agents require a Box observation space")
    if not isinstance(env.action_space, spaces.Discrete):
        raise TypeError("CartPole agents require a Discrete action space")
    return int(np.prod(env.observation_space.shape)), int(env.action_space.n)
