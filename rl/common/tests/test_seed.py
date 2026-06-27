import pytest

from common.seed import spawn_seeds


def test_spawn_seeds_is_reproducible_and_separates_streams() -> None:
    seeds = spawn_seeds(master_seed=24, runs=3)

    assert seeds == spawn_seeds(master_seed=24, runs=3)
    assert len(set(seeds)) == 3
    assert all(env_seed != agent_seed for env_seed, agent_seed in seeds)


def test_spawn_seeds_requires_a_run() -> None:
    with pytest.raises(ValueError, match="runs must be at least 1"):
        spawn_seeds(master_seed=24, runs=0)
