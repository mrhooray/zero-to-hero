from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from deep.agents.common import (
    MLP,
    ReplayBuffer,
    Transition,
    cartpole_sizes,
)
from deep.type import TrainingConfig


class SACAgent:
    name = "sac"

    def __init__(self, env: gym.Env, config: TrainingConfig) -> None:
        torch.manual_seed(config.seed)

        self.config = config
        self.observation_size, self.action_count = cartpole_sizes(env)

        self.policy = MLP(self.observation_size, config.hidden_size, self.action_count)
        self.c1 = MLP(self.observation_size, config.hidden_size, self.action_count)
        self.c2 = MLP(self.observation_size, config.hidden_size, self.action_count)
        self.target_c1 = MLP(
            self.observation_size, config.hidden_size, self.action_count
        )
        self.target_c2 = MLP(
            self.observation_size, config.hidden_size, self.action_count
        )
        self.target_c1.load_state_dict(self.c1.state_dict())
        self.target_c2.load_state_dict(self.c2.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.c1.parameters()) + list(self.c2.parameters()),
            lr=config.lr,
        )

        self.replay = ReplayBuffer(config.replay_size, config.seed)
        self.steps = 0

    def start_episode(self, episode: int) -> None:
        pass

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        with torch.no_grad():
            logits = self.policy(torch.as_tensor(observation, dtype=torch.float32))
        if training:
            return int(Categorical(logits=logits).sample().item())
        return int(torch.argmax(logits).item())

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.replay.append(
            Transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
            )
        )
        self._update()

    def end_episode(self) -> None:
        pass

    def _update(self) -> None:
        self.steps += 1
        if len(self.replay) < max(self.config.warmup_steps, self.config.batch_size):
            return

        observations, actions, rewards, next_observations, terminated = (
            self.replay.sample(self.config.batch_size)
        )
        self._update_critics(
            observations,
            actions,
            rewards,
            next_observations,
            terminated,
        )
        self._update_policy(observations)
        self._update_target_networks()

    def _update_critics(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminated: torch.Tensor,
    ) -> None:
        c1_values = self.c1(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        c2_values = self.c2(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_logits = self.policy(next_observations)
            next_probs = next_logits.softmax(dim=1)
            next_log_probs = next_logits.log_softmax(dim=1)
            next_critic_values = torch.minimum(
                self.target_c1(next_observations),
                self.target_c2(next_observations),
            )
            entropy_bonus = -self.config.sac_alpha * next_log_probs
            next_value = (next_probs * (next_critic_values + entropy_bonus)).sum(dim=1)
            target = rewards + self.config.gamma * (1.0 - terminated) * next_value

        loss = nn.functional.mse_loss(c1_values, target) + nn.functional.mse_loss(
            c2_values, target
        )
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def _update_policy(self, observations: torch.Tensor) -> None:
        logits = self.policy(observations)
        probs = logits.softmax(dim=1)
        log_probs = logits.log_softmax(dim=1)
        critic_values = torch.minimum(self.c1(observations), self.c2(observations))
        critic_values = critic_values.detach()
        entropy_penalty = self.config.sac_alpha * log_probs
        loss = (probs * (entropy_penalty - critic_values)).sum(dim=1)
        self.policy_optimizer.zero_grad()
        loss.mean().backward()
        self.policy_optimizer.step()

    def _update_target_networks(self) -> None:
        with torch.no_grad():
            for target_param, critic_param in zip(
                self.target_c1.parameters(), self.c1.parameters()
            ):
                target_param.lerp_(critic_param, self.config.sac_target_tau)
            for target_param, critic_param in zip(
                self.target_c2.parameters(), self.c2.parameters()
            ):
                target_param.lerp_(critic_param, self.config.sac_target_tau)
