from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planning.agents import AGENT_NAMES, AGENTS
from common.grid_cli import grid_main
from planning.type import TrainingConfig


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main() -> None:
    grid_main(
        description="Train and benchmark GridWorld planning agents.",
        algorithm_default="mcts",
        algorithm_choices=AGENT_NAMES,
        agent_map=AGENTS,
        config_fn=_planning_config,
    )


# -------------------------------------------------------------------------
# Wiring
# -------------------------------------------------------------------------
def _planning_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(seed=args.agent_seed)


if __name__ == "__main__":
    main()
