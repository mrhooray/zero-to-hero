from __future__ import annotations

import numpy as np


def spawn_seeds(
    master_seed: int,
    runs: int,
) -> list[tuple[int, int]]:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    env_seeds = np.random.SeedSequence(master_seed, spawn_key=(0,)).generate_state(runs)
    agent_seeds = np.random.SeedSequence(master_seed, spawn_key=(1,)).generate_state(
        runs
    )
    return [
        (int(run_env_seed), int(run_agent_seed))
        for run_env_seed, run_agent_seed in zip(env_seeds, agent_seeds)
    ]
