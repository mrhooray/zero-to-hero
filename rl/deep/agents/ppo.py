from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from deep.agents.common import MLP, Transition, cartpole_sizes
from deep.type import TrainingConfig


@dataclass(frozen=True)
class PPOConfig:
    epochs: int = 4
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gradient_clip: float = 0.5


@dataclass(frozen=True)
class RolloutStep:
    transition: Transition
    log_prob: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class PendingStep:
    log_prob: torch.Tensor
    value: torch.Tensor


class PPOAgent:
    name = "ppo"

    def __init__(
        self,
        env: gym.Env,
        config: TrainingConfig,
        ppo_config: PPOConfig = PPOConfig(),
    ) -> None:
        torch.manual_seed(config.seed)

        self.config = config
        self.ppo_config = ppo_config
        observation_size, action_count = cartpole_sizes(env)

        self.policy = MLP(observation_size, config.hidden_size, action_count)
        self.value = MLP(observation_size, config.hidden_size, 1)
        self.parameters = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=config.lr)

        self.rollout: list[RolloutStep] = []
        self.pending_step: PendingStep | None = None

    def start_episode(self, episode: int) -> None:
        self.rollout = []
        self.pending_step = None

    @torch.no_grad()
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        logits = self.policy(torch.as_tensor(observation, dtype=torch.float32))
        if not training:
            return int(torch.argmax(logits).item())

        distribution = Categorical(logits=logits)
        action = distribution.sample()
        self.pending_step = PendingStep(
            log_prob=distribution.log_prob(action),
            value=self.value(
                torch.as_tensor(observation, dtype=torch.float32)
            ).squeeze(),
        )
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
        if self.pending_step is None:
            return

        self.rollout.append(
            RolloutStep(
                transition=Transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                ),
                log_prob=self.pending_step.log_prob,
                value=self.pending_step.value,
            )
        )
        self.pending_step = None

    def end_episode(self) -> None:
        if self.rollout:
            self._update()

    def _update(self) -> None:
        observations = torch.as_tensor(
            np.array([step.transition.observation for step in self.rollout]),
            dtype=torch.float32,
        )
        actions = torch.as_tensor(
            [step.transition.action for step in self.rollout], dtype=torch.int64
        )
        old_log_probs = torch.stack([step.log_prob for step in self.rollout])
        old_values = torch.stack([step.value for step in self.rollout])
        advantages = self._advantages(old_values)
        returns = advantages + old_values

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_config.epochs):
            indices = torch.randperm(len(actions))
            for start in range(0, len(actions), self.config.batch_size):
                batch_indices = indices[start : start + self.config.batch_size]
                self._update_batch(
                    observations[batch_indices],
                    actions[batch_indices],
                    old_log_probs[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices],
                )

    def _update_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        logits = self.policy(observations)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        values = self.value(observations).squeeze(1)

        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = ratio.clamp(
            1.0 - self.ppo_config.clip_ratio,
            1.0 + self.ppo_config.clip_ratio,
        )
        policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_bonus = distribution.entropy()

        loss = (
            policy_loss.mean()
            + self.ppo_config.value_coef * value_loss
            - self.ppo_config.entropy_coef * entropy_bonus.mean()
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.ppo_config.gradient_clip)
        self.optimizer.step()

    def _advantages(self, values: torch.Tensor) -> torch.Tensor:
        advantages = []
        advantage = 0.0
        next_value = 0.0
        for index in reversed(range(len(self.rollout))):
            step = self.rollout[index]
            mask = 1.0 - float(step.transition.terminated or step.transition.truncated)
            delta = (
                step.transition.reward
                + self.config.gamma * next_value * mask
                - values[index].item()
            )
            advantage = (
                delta
                + self.config.gamma * self.ppo_config.gae_lambda * mask * advantage
            )
            advantages.append(advantage)
            next_value = values[index].item()

        advantages.reverse()
        return torch.as_tensor(advantages, dtype=torch.float32)
