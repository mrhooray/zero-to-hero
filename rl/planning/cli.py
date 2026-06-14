from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planning.agents import AGENT_NAMES, AGENTS
from planning.gridworld import GridWorldEnv
from planning.plot import plot_returns
from planning.runner import evaluate, train
from planning.type import TrainingConfig


# -------------------------------------------------------------------------
# Entry point and parser
# -------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and benchmark GridWorld planning agents."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render")
    add_env_args(render_parser)
    render_parser.set_defaults(func=render_command)

    train_parser = subparsers.add_parser("train")
    add_env_args(train_parser)
    add_runner_args(train_parser)
    train_parser.add_argument("--algorithm", default="mcts", choices=AGENT_NAMES)
    train_parser.add_argument("--plot", help="write a reward curve PNG")
    train_parser.set_defaults(func=train_command)

    benchmark_parser = subparsers.add_parser("benchmark")
    add_env_args(benchmark_parser)
    add_runner_args(benchmark_parser)
    benchmark_parser.add_argument("--plot", help="write all reward curves to one PNG")
    benchmark_parser.set_defaults(func=benchmark_command)

    return parser


# -------------------------------------------------------------------------
# Commands
# -------------------------------------------------------------------------
def render_command(args: argparse.Namespace) -> None:
    env = env_from_args(args)
    env.reset()
    print(env.render())


def train_command(args: argparse.Namespace) -> None:
    config = training_config_from_args(args)
    env = env_from_args(args)
    agent = AGENTS[args.algorithm](env, config)
    train_result = train(env, agent, episodes=args.train_episodes)
    eval_result = evaluate(
        env_from_args(args),
        agent,
        episodes=args.eval_episodes,
    )

    print(
        f"trained {train_result.agent_name} for {len(train_result.episodes)} episodes"
    )
    print(f"train success rate: {train_result.success_rate():.2%}")
    print(f"eval mean return: {eval_result.mean_return():.3f}")
    print(f"eval success rate: {eval_result.success_rate():.2%}")
    if args.plot:
        plot_returns([train_result], args.plot)
        print(f"wrote {args.plot}")


def benchmark_command(args: argparse.Namespace) -> None:
    config = training_config_from_args(args)
    train_results = []
    eval_results = []
    print(f"{'agent':<22} {'mean_return':>12} {'success':>10} {'mean_len':>10}")
    for name in AGENT_NAMES:
        env = env_from_args(args)
        agent = AGENTS[name](env, config)
        train_results.append(train(env, agent, episodes=args.train_episodes))
        eval_results.append(
            evaluate(env_from_args(args), agent, episodes=args.eval_episodes)
        )

    for row in eval_results:
        print(
            f"{row.agent_name:<22} "
            f"{row.mean_return():>12.3f} "
            f"{row.success_rate():>9.2%} "
            f"{row.mean_length():>10.1f}"
        )
    if args.plot:
        plot_returns(train_results, args.plot, show_raw=False)
        print(f"wrote {args.plot}")


# -------------------------------------------------------------------------
# Argument helpers
# -------------------------------------------------------------------------
def add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=24)


def add_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-episodes", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=32)


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(seed=args.seed)


def env_from_args(args: argparse.Namespace) -> GridWorldEnv:
    start = (0, 0)
    goal = (args.size - 1, args.size - 1)
    return GridWorldEnv(
        size=args.size,
        start=start,
        traps=GridWorldEnv.generate_traps(args.size, args.seed, start, goal),
    )


if __name__ == "__main__":
    main()
