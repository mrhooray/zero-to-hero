from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.grid_world import GridWorldEnv
from tabular.agents import AGENT_NAMES, AGENTS
from common.grid_cli import grid_main
from tabular.type import TrainingConfig


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main() -> None:
    grid_main(
        description="Train and benchmark GridWorld RL agents.",
        algorithm_default="q-learning",
        algorithm_choices=AGENT_NAMES,
        agent_map=AGENTS,
        config_fn=_tabular_config,
        extra_train_args=_extra_train_args,
        extra_train_fn=_extra_train_fn,
    )


# -------------------------------------------------------------------------
# Wiring
# -------------------------------------------------------------------------
def _tabular_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epsilon_decay_episodes=max(1, int(args.train_episodes * 0.5)),
        seed=args.seed,
    )


def _extra_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--show-policy", action="store_true")


def _extra_train_fn(args: argparse.Namespace, agent: object, env: GridWorldEnv) -> None:
    if args.show_policy and hasattr(agent, "q_values"):
        print()
        print(env.render_policy(agent.q_values))


if __name__ == "__main__":
    main()
