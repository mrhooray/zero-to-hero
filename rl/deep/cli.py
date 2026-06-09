from __future__ import annotations

import argparse

from pathlib import Path
import sys

import gymnasium as gym

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep.agents import AGENT_NAMES, AGENTS
from deep.plot import plot_returns
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
    add_runner_args(train_parser)
    add_training_config_args(train_parser)
    train_parser.add_argument("--algorithm", default="dqn", choices=AGENT_NAMES)
    train_parser.add_argument("--plot", help="write a reward curve PNG")
    train_parser.set_defaults(func=train_command)

    benchmark_parser = subparsers.add_parser("benchmark")
    add_runner_args(benchmark_parser)
    add_training_config_args(benchmark_parser)
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
        seed=args.seed,
    )
    eval_result = evaluate(
        make_env(), agent, episodes=args.eval_episodes, seed=args.seed
    )

    print(
        f"trained {train_result.agent_name} for {len(train_result.episodes)} episodes"
    )
    print(f"train mean return: {train_result.mean_return():.1f}")
    print(f"train mean length: {train_result.mean_length():.1f}")
    print(f"train termination rate: {train_result.termination_rate():.2%}")
    print(f"train success rate: {train_result.success_rate():.2%}")
    print(
        f"evaluated {eval_result.agent_name} for {len(eval_result.episodes)} episodes"
    )
    print(f"eval mean return: {eval_result.mean_return():.1f}")
    print(f"eval mean length: {eval_result.mean_length():.1f}")
    print(f"eval termination rate: {eval_result.termination_rate():.2%}")
    print(f"eval success rate: {eval_result.success_rate():.2%}")
    if args.plot:
        plot_returns([train_result], args.plot)
        print(f"wrote {args.plot}")


def benchmark_command(args: argparse.Namespace) -> None:
    config = training_config_from_args(args)
    train_results = []
    print(
        f"{'agent':<12} "
        f"{'train_return':>13} "
        f"{'train_len':>10} "
        f"{'train_term':>10} "
        f"{'train_success':>13} "
        f"{'eval_return':>12} "
        f"{'eval_len':>10} "
        f"{'eval_term':>10} "
        f"{'eval_success':>12}"
    )
    for name in AGENT_NAMES:
        env = make_env()
        agent = AGENTS[name](env, config)
        train_result = train(
            env,
            agent,
            episodes=args.train_episodes,
            seed=args.seed,
        )
        train_results.append(train_result)
        eval_result = evaluate(
            make_env(),
            agent,
            episodes=args.eval_episodes,
            seed=args.seed,
        )
        print(
            f"{name:<12} "
            f"{train_result.mean_return():>13.1f} "
            f"{train_result.mean_length():>10.1f} "
            f"{train_result.termination_rate():>9.2%} "
            f"{train_result.success_rate():>12.2%} "
            f"{eval_result.mean_return():>12.1f} "
            f"{eval_result.mean_length():>10.1f} "
            f"{eval_result.termination_rate():>9.2%} "
            f"{eval_result.success_rate():>11.2%}"
        )
    if args.plot:
        plot_returns(train_results, args.plot, show_raw=False)
        print(f"wrote {args.plot}")


# -------------------------------------------------------------------------
# Argument helpers
# -------------------------------------------------------------------------
def add_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=24)
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
        seed=args.seed,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        dqn_epsilon_decay_episodes=max(1, int(args.train_episodes * 0.5)),
        debug=args.debug,
    )


def make_env() -> gym.Env:
    return gym.make("CartPole-v1")


if __name__ == "__main__":
    main()
