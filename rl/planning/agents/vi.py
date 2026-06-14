from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from common.grid_world import (
    GridWorldEnv,
    decode_position,
    encode_position,
    grid_transition,
)
from planning.type import TrainingConfig


class ValueIterationAgent:
    name = "value-iteration"

    def __init__(
        self,
        env: gym.Env[int, int],
        config: TrainingConfig,
        theta: float = 1e-6,
    ) -> None:
        if not isinstance(env, GridWorldEnv):
            raise TypeError("ValueIterationAgent requires GridWorldEnv")

        self.config = config
        self.size = env.size
        self.goal = env.goal
        self.traps = set(env.traps)
        self.action_count = int(env.action_space.n)
        self.state_count = int(env.observation_space.n)
        self.terminal_observations = {
            encode_position(self.size, self.goal),
            *(encode_position(self.size, trap) for trap in self.traps),
        }

        self._solve(theta)

    def start_episode(self, episode: int) -> None:
        pass

    def select_action(self, observation: int, training: bool = True) -> int:
        return int(self.policy[observation])

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        pass

    def end_episode(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # Value iteration
    # -------------------------------------------------------------------------

    def _solve(self, theta: float) -> None:
        v = np.zeros(self.state_count, dtype=np.float64)
        policy = np.zeros(self.state_count, dtype=np.int64)

        while True:
            delta = 0.0
            for obs in range(self.state_count):
                if obs in self.terminal_observations:
                    continue
                best_value = -math.inf
                best_action = int(policy[obs])
                position = decode_position(self.size, obs)
                for action in range(self.action_count):
                    next_pos, reward, terminated = grid_transition(
                        self.size,
                        self.goal,
                        self.traps,
                        position,
                        action,
                    )
                    next_obs = encode_position(self.size, next_pos)
                    value = reward
                    if not terminated:
                        value += self.config.gamma * v[next_obs]
                    if value > best_value:
                        best_value = value
                        best_action = action
                delta = max(delta, abs(best_value - v[obs]))
                v[obs] = best_value
                policy[obs] = best_action
            if delta < theta:
                break

        self.v = v
        self.policy = policy
