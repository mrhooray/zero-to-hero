from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def cartpole_sizes(env: gym.Env) -> tuple[int, int]:
    if not isinstance(env.observation_space, spaces.Box):
        raise TypeError("CartPole agents require a Box observation space")
    if not isinstance(env.action_space, spaces.Discrete):
        raise TypeError("CartPole agents require a Discrete action space")
    return int(np.prod(env.observation_space.shape)), int(env.action_space.n)
