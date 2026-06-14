import pytest

from common.grid_world import GridWorldEnv


def test_gridworld_reaches_goal() -> None:
    env = GridWorldEnv(size=3)
    observation, info = env.reset()

    assert observation == env.encode((0, 0))
    assert info["position"] == (0, 0)

    for action in [1, 1, 2, 2]:
        observation, reward, terminated, truncated, info = env.step(action)

    assert observation == env.encode((2, 2))
    assert reward == pytest.approx(1.0)
    assert terminated
    assert not truncated
    assert info["is_success"]


def test_grid_boundary_blocks_movement() -> None:
    env = GridWorldEnv(size=3)
    observation, _ = env.reset()
    observation, _, _, _, info = env.step(0)

    assert observation == env.encode((0, 0))
    assert info["position"] == (0, 0)


def test_seeded_trap_generation_is_deterministic_and_solvable() -> None:
    start = (0, 0)
    goal = (7, 7)
    traps = GridWorldEnv.generate_traps(size=8, seed=24, start=start, goal=goal)

    assert traps == GridWorldEnv.generate_traps(size=8, seed=24, start=start, goal=goal)
    assert len(traps) == 4
    assert start not in traps
    assert goal not in traps
    GridWorldEnv(size=8, start=start, goal=goal, traps=traps)
