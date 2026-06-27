from __future__ import annotations

import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep.agents import AGENT_NAMES, AGENTS
from common.plot import plot_returns
from common.seed import spawn_seeds
from common.type import RunResult
from deep.runner import evaluate, train
from deep.type import TrainingConfig


# -------------------------------------------------------------------------
# Entry point and parser
# -------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and benchmark CartPole deep RL agents."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_episode_args(train_parser)
    add_training_config_args(train_parser)
    train_parser.add_argument("--env-seed", type=int, default=24)
    train_parser.add_argument("--agent-seed", type=int, default=24)
    train_parser.add_argument("--algorithm", default="dqn", choices=AGENT_NAMES)
    train_parser.add_argument("--plot", help="write a reward curve PNG")
    train_parser.set_defaults(func=train_command)

    benchmark_parser = subparsers.add_parser("benchmark")
    add_episode_args(benchmark_parser)
    add_training_config_args(benchmark_parser)
    benchmark_parser.add_argument("--seed", type=int, default=24)
    benchmark_parser.add_argument("--runs", type=int, default=8)
    benchmark_parser.add_argument("--plot", help="write all reward curves to one PNG")
    benchmark_parser.set_defaults(func=benchmark_command)

    return parser


# -------------------------------------------------------------------------
# Commands
# -------------------------------------------------------------------------
def train_command(args: argparse.Namespace) -> None:
    config = training_config_from_args(args)
    env = make_env()
    agent = AGENTS[args.algorithm](env, config)
    train_result = train(
        env,
        agent,
        episodes=args.train_episodes,
        seed=args.env_seed,
    )
    # Training used seeds [env_seed, env_seed + train_episodes).
    eval_result = evaluate(
        make_env(),
        agent,
        episodes=args.eval_episodes,
        seed=args.env_seed + args.train_episodes,
    )

    print(
        f"trained {train_result.agent_name} for {len(train_result.episodes)} episodes"
    )
    print(f"train mean return: {train_result.mean_return():.1f}")
    print(f"train termination rate: {train_result.termination_rate():.2%}")
    print(f"train success rate: {train_result.success_rate():.2%}")
    print()
    print(
        f"evaluated {eval_result.agent_name} for {len(eval_result.episodes)} episodes"
    )
    print(f"eval mean return: {eval_result.mean_return():.1f}")
    print(f"eval termination rate: {eval_result.termination_rate():.2%}")
    print(f"eval success rate: {eval_result.success_rate():.2%}")
    if args.plot:
        plot_returns([[train_result]], args.plot)
        print(f"learning curves plot saved to {args.plot}")


def benchmark_command(args: argparse.Namespace) -> None:
    run_seeds = spawn_seeds(args.seed, args.runs)
    benchmark_results: list[list[RunResult]] = []
    row_format = "{:<12}  {:>13}  {:>14}  {:>14}"
    print(f"{'run':>3}  {'env_seed':>10}  {'agent_seed':>10}")
    for run, (env_seed, agent_seed) in enumerate(run_seeds):
        print(f"{run:>3}  {env_seed:>10}  {agent_seed:>10}")
    print()
    print(
        row_format.format(
            "agent",
            "eval_return",
            "eval_term",
            "eval_success",
        )
    )
    for name in AGENT_NAMES:
        train_results = []
        eval_results = []
        for env_seed, agent_seed in run_seeds:
            run_args = argparse.Namespace(**{**vars(args), "agent_seed": agent_seed})
            env = make_env()
            agent = AGENTS[name](env, training_config_from_args(run_args))
            train_results.append(
                train(
                    env,
                    agent,
                    episodes=args.train_episodes,
                    seed=env_seed,
                )
            )
            # Training used seeds [env_seed, env_seed + train_episodes).
            eval_result = evaluate(
                make_env(),
                agent,
                episodes=args.eval_episodes,
                seed=env_seed + args.train_episodes,
            )
            eval_results.append(eval_result)

        benchmark_results.append(train_results)
        print(
            row_format.format(
                name,
                _mean_std(eval_results, "mean_return"),
                _mean_std(eval_results, "termination_rate", percent=True),
                _mean_std(eval_results, "success_rate", percent=True),
            )
        )
    if args.plot:
        plot_returns(benchmark_results, args.plot)
        print(f"learning curves plot saved to {args.plot}")


# -------------------------------------------------------------------------
# Argument helpers
# -------------------------------------------------------------------------
def add_episode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-episodes", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=32)


def add_training_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=1024 * 2)
    parser.add_argument("--debug", action="store_true")


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        lr=args.lr,
        seed=args.agent_seed,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        debug=args.debug,
    )


def make_env() -> gym.Env:
    return gym.make("CartPole-v1")


def _mean_std(
    results: list[RunResult],
    metric: str,
    percent: bool = False,
) -> str:
    values = np.array([getattr(result, metric)() for result in results])
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{np.mean(values) * scale:.1f} ± {np.std(values) * scale:.1f}{suffix}"


if __name__ == "__main__":
    main()
