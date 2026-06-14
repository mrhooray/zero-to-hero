from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import gymnasium as gym
import numpy as np

from common.grid_world import (
    GridWorldEnv,
    Position,
    decode_position,
    grid_transition,
    neighbors,
    next_position,
)
from planning.type import TrainingConfig


@dataclass(frozen=True)
class MCTSConfig:
    search_depth: int | None = None
    simulation_count: int = 16
    exploration_weight: float = math.sqrt(2.0)
    rollout_epsilon: float = 0.1


class MCTSAgent:
    name = "mcts"

    def __init__(
        self,
        env: gym.Env[int, int],
        config: TrainingConfig,
        mcts_config: MCTSConfig = MCTSConfig(),
    ) -> None:
        if not isinstance(env, GridWorldEnv):
            raise TypeError("MCTSAgent requires GridWorldEnv")
        if mcts_config.simulation_count < 1:
            raise ValueError("simulation_count must be at least 1")
        if not 0.0 <= mcts_config.rollout_epsilon <= 1.0:
            raise ValueError("rollout_epsilon must be between 0 and 1")

        self.config = config
        self.mcts_config = mcts_config
        self.size = env.size
        self.goal = env.goal
        self.traps = set(env.traps)

        self.search_depth = (
            mcts_config.search_depth
            if mcts_config.search_depth is not None
            else env.size
        )
        self.simulation_count = mcts_config.simulation_count
        self.exploration_weight = mcts_config.exploration_weight
        self.rollout_epsilon = mcts_config.rollout_epsilon

        self.goal_distances = self._goal_distances()
        self.action_count = int(env.action_space.n)
        self.rng = np.random.default_rng(config.seed)

    def start_episode(self, episode: int) -> None:
        pass

    def select_action(self, observation: int, training: bool = True) -> int:
        root = _Node(position=decode_position(self.size, observation), depth=0)
        root.untried_actions = self._candidate_actions(root.position)

        for _ in range(self.simulation_count):
            self._search(root)

        return self._select_best_action(root)

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
    # Search
    # -------------------------------------------------------------------------
    def _search(self, root: _Node) -> None:
        path = self._select_path(root)
        _, node = path[-1]
        value = 0.0
        if not node.terminal and node.depth < self.search_depth:
            edge = self._expand_one_child(node)
            path.append((edge, edge.child))
            value = self._rollout(edge.child)

        self._backprop(path, value)

    def _select_path(self, root: _Node) -> list[tuple[_Edge | None, _Node]]:
        path = [(None, root)]
        node = root
        while (
            not node.terminal
            and node.depth < self.search_depth
            and not node.untried_actions
            and node.edges
        ):
            edge = self._select_uct_edge(node)
            node = edge.child
            path.append((edge, node))
        return path

    def _select_uct_edge(self, node: _Node) -> _Edge:
        scores = [
            (
                self._uct_score(node, edge),
                edge.action,
                edge,
            )
            for edge in node.edges.values()
        ]
        best_score = max(score for score, _, _ in scores)
        candidates = [
            edge for score, _, edge in scores if math.isclose(score, best_score)
        ]
        return candidates[int(self.rng.integers(len(candidates)))]

    def _uct_score(self, node: _Node, edge: _Edge) -> float:
        child_visits = max(1, edge.child.visits)
        value = edge.mean_return(self.config.gamma)
        exploration = self.exploration_weight * math.sqrt(
            math.log(max(1, node.visits)) / child_visits
        )
        return value + exploration

    def _expand_one_child(self, node: _Node) -> _Edge:
        action_index = int(self.rng.integers(len(node.untried_actions)))
        action = node.untried_actions.pop(action_index)
        next_position, reward, terminal = grid_transition(
            self.size,
            self.goal,
            self.traps,
            node.position,
            action,
        )
        edge = _Edge(
            action=action,
            reward=reward,
            child=_Node(
                position=next_position,
                depth=node.depth + 1,
                terminal=terminal,
                untried_actions=self._candidate_actions(next_position),
            ),
        )
        node.edges[action] = edge
        return edge

    def _rollout(self, node: _Node) -> float:
        value = 0.0
        discount = 1.0
        position = node.position
        depth = node.depth
        terminal = node.terminal
        while not terminal and depth < self.search_depth:
            action = self._select_rollout_action(position)
            position, reward, terminal = grid_transition(
                self.size,
                self.goal,
                self.traps,
                position,
                action,
            )
            value += discount * reward
            discount *= self.config.gamma
            depth += 1
        return value

    def _backprop(
        self,
        path: list[tuple[_Edge | None, _Node]],
        value: float,
    ) -> None:
        for index in range(len(path) - 1, -1, -1):
            edge, node = path[index]
            node.visits += 1
            node.value_sum += value
            if index == 0:
                break

            if edge is None:
                raise ValueError("non-root path step must have an incoming edge")
            value = edge.reward + self.config.gamma * value

    def _select_best_action(self, root: _Node) -> int:
        edges = self._forward_edges(root)
        values = [
            (edge.mean_return(self.config.gamma), edge.child.visits, edge.action)
            for edge in edges
        ]
        best_value = max(value for value, _, _ in values)
        candidates = [
            (visits, action)
            for value, visits, action in values
            if math.isclose(value, best_value)
        ]
        best_visits = max(visits for visits, _ in candidates)
        best_actions = [
            action for visits, action in candidates if visits == best_visits
        ]
        return best_actions[int(self.rng.integers(len(best_actions)))]

    # -------------------------------------------------------------------------
    # GridWorld heuristics
    # -------------------------------------------------------------------------
    def _candidate_actions(self, position: Position) -> list[int]:
        safe_actions = []
        moving_actions = []
        for action in range(self.action_count):
            new_position = next_position(self.size, position, action)
            if new_position == position:
                continue
            moving_actions.append(action)
            if new_position not in self.traps:
                safe_actions.append(action)
        return safe_actions or moving_actions

    def _select_rollout_action(self, position: Position) -> int:
        if self.rng.random() < self.rollout_epsilon:
            return int(self.rng.integers(self.action_count))

        scores = []
        for action in self._candidate_actions(position):
            new_position = next_position(self.size, position, action)
            distance = self._goal_distance(new_position)
            scores.append((distance, action))

        if not scores:
            return int(self.rng.integers(self.action_count))

        best_score = min(score for score, _ in scores)
        candidates = [
            action for score, action in scores if math.isclose(score, best_score)
        ]
        return candidates[int(self.rng.integers(len(candidates)))]

    def _forward_edges(self, root: _Node) -> list[_Edge]:
        current_distance = self._goal_distance(root.position)
        edges = [
            edge
            for edge in root.edges.values()
            if self._goal_distance(edge.child.position) < current_distance
        ]
        return edges or list(root.edges.values())

    def _goal_distance(self, position: Position) -> int:
        return self.goal_distances.get(position, self.size * self.size)

    def _goal_distances(self) -> dict[Position, int]:
        distances = {self.goal: 0}
        queue = deque([self.goal])
        while queue:
            position = queue.popleft()
            for neighbor in neighbors(self.size, position):
                if neighbor in distances or neighbor in self.traps:
                    continue
                distances[neighbor] = distances[position] + 1
                queue.append(neighbor)
        return distances


@dataclass
class _Node:
    position: Position
    depth: int
    terminal: bool = False
    visits: int = 0
    value_sum: float = 0.0
    untried_actions: list[int] = field(default_factory=list)
    edges: dict[int, _Edge] = field(default_factory=dict)

    def mean_value(self) -> float:
        return self.value_sum / max(1, self.visits)


@dataclass
class _Edge:
    action: int
    reward: float
    child: _Node

    def mean_return(self, gamma: float) -> float:
        return self.reward + gamma * self.child.mean_value()
