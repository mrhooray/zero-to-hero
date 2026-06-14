from __future__ import annotations

from collections import deque
from random import Random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Position = tuple[int, int]

ACTION_TO_DELTA: dict[int, Position] = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

ACTION_ARROWS = {
    0: "^",
    1: ">",
    2: "v",
    3: "<",
}

STEP_REWARD = -0.05
GOAL_REWARD = 1.0
TRAP_REWARD = -1.0
TRAP_CELL_RATIO = 16


class GridWorldEnv(gym.Env[int, int]):
    # -------------------------------------------------------------------------
    # Gymnasium interface
    # -------------------------------------------------------------------------
    def __init__(
        self,
        size: int = 8,
        start: Position = (0, 0),
        goal: Position | None = None,
        traps: tuple[Position, ...] = (),
        max_steps: int | None = None,
    ) -> None:
        if size < 2:
            raise ValueError("size must be at least 2")

        self.size = size
        self.start = start
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.traps = set(traps)
        self.max_steps = max_steps if max_steps is not None else size * size * 4

        self._validate_position(self.start, "start")
        self._validate_position(self.goal, "goal")
        for trap in self.traps:
            self._validate_position(trap, "trap")
        if self.start in self.traps or self.goal in self.traps:
            raise ValueError("start and goal cannot be traps")
        if not has_path(self.size, self.start, self.goal, self.traps):
            raise ValueError("traps must leave a path from start to goal")

        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)
        self.agent_position = self.start
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self.agent_position = self.start
        self.steps = 0
        return self.encode(self.agent_position), self._info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        self.steps += 1
        new_position, reward, terminated = grid_transition(
            self.size,
            self.goal,
            self.traps,
            self.agent_position,
            action,
        )
        self.agent_position = new_position
        truncated = self.steps >= self.max_steps and not terminated

        return self.encode(new_position), reward, terminated, truncated, self._info()

    # -------------------------------------------------------------------------
    # Trap generation
    # -------------------------------------------------------------------------
    @staticmethod
    def generate_traps(
        size: int,
        seed: int,
        start: Position,
        goal: Position,
    ) -> tuple[Position, ...]:
        target_count = size * size // TRAP_CELL_RATIO
        candidates = [
            (row, col)
            for row in range(size)
            for col in range(size)
            if (row, col) not in {start, goal}
        ]
        Random(seed).shuffle(candidates)

        traps: set[Position] = set()
        for candidate in candidates:
            if len(traps) >= target_count:
                break
            next_traps = traps | {candidate}
            if has_path(size, start, goal, next_traps):
                traps = next_traps
        return tuple(sorted(traps))

    # -------------------------------------------------------------------------
    # Renderers
    # -------------------------------------------------------------------------
    def render(self) -> str:
        rows: list[str] = []
        for row in range(self.size):
            values = []
            for col in range(self.size):
                position = (row, col)
                if position == self.agent_position:
                    values.append("A")
                elif position == self.start:
                    values.append("S")
                elif position == self.goal:
                    values.append("G")
                elif position in self.traps:
                    values.append("X")
                else:
                    values.append(".")
            rows.append(" ".join(values))
        return "\n".join(rows)

    def render_policy(self, q_values: np.ndarray) -> str:
        rows: list[str] = []
        for row in range(self.size):
            values = []
            for col in range(self.size):
                position = (row, col)
                if position == self.goal:
                    values.append("G")
                elif position == self.start:
                    values.append("S")
                elif position in self.traps:
                    values.append("X")
                else:
                    action = int(np.argmax(q_values[self.encode(position)]))
                    values.append(ACTION_ARROWS[action])
            rows.append(" ".join(values))
        return "\n".join(rows)

    # -------------------------------------------------------------------------
    # Grid mechanics and utilities
    # -------------------------------------------------------------------------
    def encode(self, position: Position) -> int:
        return encode_position(self.size, position)

    def _info(self) -> dict[str, Any]:
        return {
            "position": self.agent_position,
            "steps": self.steps,
            "distance_to_goal": abs(self.agent_position[0] - self.goal[0])
            + abs(self.agent_position[1] - self.goal[1]),
            "is_success": self.agent_position == self.goal,
        }

    def _validate_position(self, position: Position, name: str) -> None:
        row, col = position
        if not 0 <= row < self.size or not 0 <= col < self.size:
            raise ValueError(f"{name} position {position} is outside the grid")


def grid_transition(
    size: int,
    goal: Position,
    traps: set[Position],
    position: Position,
    action: int,
) -> tuple[Position, float, bool]:
    new_position = next_position(size, position, action)
    if new_position == goal:
        return new_position, GOAL_REWARD, True
    if new_position in traps:
        return new_position, TRAP_REWARD, True
    return new_position, STEP_REWARD, False


def encode_position(size: int, position: Position) -> int:
    row, col = position
    return row * size + col


def decode_position(size: int, observation: int) -> Position:
    return divmod(observation, size)


def next_position(size: int, position: Position, action: int) -> Position:
    if action not in ACTION_TO_DELTA:
        raise ValueError(f"invalid action: {action}")

    row_delta, col_delta = ACTION_TO_DELTA[action]
    row, col = position
    return (
        min(max(row + row_delta, 0), size - 1),
        min(max(col + col_delta, 0), size - 1),
    )


def has_path(
    size: int,
    start: Position,
    goal: Position,
    traps: set[Position],
) -> bool:
    queue = deque([start])
    visited = {start}

    while queue:
        position = queue.popleft()
        if position == goal:
            return True
        for neighbor in neighbors(size, position):
            if neighbor in visited or neighbor in traps:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return False


def neighbors(size: int, position: Position) -> list[Position]:
    row, col = position
    values = []
    for row_delta, col_delta in ACTION_TO_DELTA.values():
        candidate = (row + row_delta, col + col_delta)
        if 0 <= candidate[0] < size and 0 <= candidate[1] < size:
            values.append(candidate)
    return values
