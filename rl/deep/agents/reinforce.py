from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from deep.agents.common import MLP, cartpole_sizes
from deep.type import TrainingConfig


class ReinforceAgent:
    name = "reinforce"

    def __init__(self, env: gym.Env, config: TrainingConfig) -> None:
        torch.manual_seed(config.seed)

        self.config = config
        observation_size, action_count = cartpole_sizes(env)

        self.policy = MLP(observation_size, config.hidden_size, action_count)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []

    def start_episode(self, episode: int) -> None:
        self.log_probs = []
        self.rewards = []

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        logits = self.policy(torch.as_tensor(observation, dtype=torch.float32))
        if not training:
            with torch.no_grad():
                return int(torch.argmax(logits).item())

        distribution = Categorical(logits=logits)
        action = distribution.sample()
        self.log_probs.append(distribution.log_prob(action))
        return int(action.item())

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.rewards.append(reward)

    def end_episode(self) -> None:
        if self.rewards:
            self._update()

    def _update(self) -> None:
        returns = []
        value = 0.0
        for reward in reversed(self.rewards):
            value = reward + self.config.gamma * value
            returns.append(value)
        returns.reverse()

        returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
        returns_tensor = returns_tensor - returns_tensor.mean()

        loss = -torch.stack(self.log_probs).mul(returns_tensor).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
