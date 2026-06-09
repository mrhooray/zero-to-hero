from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from deep.agents.common import (
    MLP,
    ReplayBuffer,
    Transition,
    cartpole_sizes,
)
from deep.type import TrainingConfig


class DQNAgent:
    name = "dqn"

    def __init__(self, env: gym.Env, config: TrainingConfig) -> None:
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)

        self.config = config
        self.observation_size, self.action_count = cartpole_sizes(env)

        self.q = MLP(self.observation_size, config.hidden_size, self.action_count)
        self.target_q = MLP(
            self.observation_size, config.hidden_size, self.action_count
        )
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.q.parameters(),
            lr=config.lr,
            amsgrad=True,
        )

        self.steps = 0
        self.episode = 0
        self.epsilon = config.dqn_epsilon_start
        self.replay = ReplayBuffer(config.replay_size, config.seed)

    def start_episode(self, episode: int) -> None:
        self.episode = episode
        self.epsilon = _epsilon_for_episode(self.config, episode)

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))

        with torch.no_grad():
            q_values = self.q(torch.as_tensor(observation, dtype=torch.float32))
        return int(torch.argmax(q_values).item())

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
        q_values = self.q(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # DQN: target_q chooses and evaluates the next action.
            # next_q = self.target_q(next_observations).max(dim=1).values

            # Double DQN: q chooses the next action, target_q evaluates it.
            next_actions = self.q(next_observations).argmax(dim=1)
            next_q = (
                self.target_q(next_observations)
                .gather(
                    1,
                    next_actions.unsqueeze(1),
                )
                .squeeze(1)
            )
            targets = rewards + self.config.gamma * (1.0 - terminated) * next_q

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.q.parameters(), self.config.dqn_gradient_clip)
        self.optimizer.step()
        self._update_target_network()

        if self.config.debug and self.steps % 1000 == 0:
            print(
                f"episode={self.episode} "
                f"step={self.steps} "
                f"q: {self._value_summary(q_values)} "
                f"target: {self._value_summary(targets)} "
                f"gradients: {self._gradient_summary()}"
            )

    def _update_target_network(self) -> None:
        with torch.no_grad():
            for target_param, q_param in zip(
                self.target_q.parameters(), self.q.parameters()
            ):
                target_param.lerp_(q_param, self.config.dqn_target_tau)

    def _value_summary(self, values: torch.Tensor) -> str:
        values = values.detach()
        return (
            f"mean={values.mean().item():.2f} "
            f"p95={values.quantile(0.95).item():.2f} "
            f"max={values.max().item():.2f}"
        )

    def _gradient_summary(self) -> str:
        gradients = torch.cat(
            [
                parameter.grad.detach().abs().flatten()
                for parameter in self.q.parameters()
                if parameter.grad is not None
            ]
        )
        return (
            f"mean={gradients.mean().item():.3f} "
            f"p95={gradients.quantile(0.95).item():.3f} "
            f"max={gradients.max().item():.3f}"
        )


def _epsilon_for_episode(config: TrainingConfig, episode: int) -> float:
    decay_episodes = max(1, config.dqn_epsilon_decay_episodes)
    progress = min(episode / decay_episodes, 1.0)
    return config.dqn_epsilon_start + progress * (
        config.dqn_epsilon_end - config.dqn_epsilon_start
    )
