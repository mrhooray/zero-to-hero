from __future__ import annotations

from typing import Protocol


class Agent(Protocol):
    name: str

    def start_episode(self, episode: int) -> None: ...

    def select_action(self, observation: int, training: bool = True) -> int: ...

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        terminated: bool,
        truncated: bool,
    ) -> None: ...

    def end_episode(self) -> None: ...
