from __future__ import annotations

import argparse
from collections.abc import Callable

from common.grid_runner import evaluate, train
from common.grid_world import GridWorldEnv
from common.plot import plot_returns
from common.type import RunResult


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def grid_main(
    description: str,
    algorithm_default: str,
    algorithm_choices: tuple[str, ...] | list[str],
    agent_map: dict[str, type],
    config_fn: Callable[[argparse.Namespace], object],
    extra_train_args: Callable[[argparse.ArgumentParser], None] | None = None,
    extra_train_fn: Callable[[argparse.Namespace, object, GridWorldEnv], None]
    | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render")
    add_env_args(render_parser)
    render_parser.set_defaults(func=_render)

    train_parser = subparsers.add_parser("train")
    add_env_args(train_parser)
    add_runner_args(train_parser)
    train_parser.add_argument(
        "--algorithm", default=algorithm_default, choices=algorithm_choices
    )
    train_parser.add_argument("--plot", help="write a reward curve PNG")
    if extra_train_args:
        extra_train_args(train_parser)
    train_parser.set_defaults(
        func=lambda args: _train(args, config_fn, agent_map, extra_train_fn)
    )

    benchmark_parser = subparsers.add_parser("benchmark")
    add_env_args(benchmark_parser)
    add_runner_args(benchmark_parser)
    benchmark_parser.add_argument("--plot", help="write all reward curves to one PNG")
    benchmark_parser.set_defaults(
        func=lambda args: _benchmark(args, config_fn, agent_map, algorithm_choices)
    )

    args = parser.parse_args()
    args.func(args)


# -------------------------------------------------------------------------
# Commands
# -------------------------------------------------------------------------
def _render(args: argparse.Namespace) -> None:
    env = env_from_args(args)
    env.reset()
    print(env.render())


def _train(
    args: argparse.Namespace,
    config_fn: Callable[[argparse.Namespace], object],
    agent_map: dict[str, type],
    extra_train_fn: Callable[[argparse.Namespace, object, GridWorldEnv], None] | None,
) -> None:
    config = config_fn(args)
    env = env_from_args(args)
    agent = agent_map[args.algorithm](env, config)
    train_result = train(env, agent, episodes=args.train_episodes)
    eval_result = evaluate(env_from_args(args), agent, episodes=args.eval_episodes)

    print_train_eval_summary(train_result, eval_result)
    if args.plot:
        plot_returns([train_result], args.plot)
        print(f"wrote {args.plot}")
    if extra_train_fn:
        extra_train_fn(args, agent, env)


def _benchmark(
    args: argparse.Namespace,
    config_fn: Callable[[argparse.Namespace], object],
    agent_map: dict[str, type],
    agent_names: tuple[str, ...] | list[str],
) -> None:
    config = config_fn(args)
    train_results: list[RunResult] = []
    eval_results: list[RunResult] = []
    print_grid_benchmark_header()
    for name in agent_names:
        env = env_from_args(args)
        agent = agent_map[name](env, config)
        train_results.append(train(env, agent, episodes=args.train_episodes))
        eval_results.append(
            evaluate(env_from_args(args), agent, episodes=args.eval_episodes)
        )

    print_grid_benchmark_rows(eval_results)
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


def env_from_args(args: argparse.Namespace) -> GridWorldEnv:
    start = (0, 0)
    goal = (args.size - 1, args.size - 1)
    return GridWorldEnv(
        size=args.size,
        start=start,
        traps=GridWorldEnv.generate_traps(args.size, args.seed, start, goal),
    )


# -------------------------------------------------------------------------
# Print helpers
# -------------------------------------------------------------------------
def print_train_eval_summary(train_result: RunResult, eval_result: RunResult) -> None:
    print(
        f"trained {train_result.agent_name} for {len(train_result.episodes)} episodes"
    )
    print(f"train success rate: {train_result.success_rate():.2%}")
    print(f"eval mean return: {eval_result.mean_return():.3f}")
    print(f"eval success rate: {eval_result.success_rate():.2%}")


def print_grid_benchmark_header() -> None:
    print(f"{'agent':<22} {'mean_return':>12} {'success':>10} {'mean_len':>10}")


def print_grid_benchmark_rows(results: list[RunResult]) -> None:
    for row in results:
        print(
            f"{row.agent_name:<22} "
            f"{row.mean_return():>12.3f} "
            f"{row.success_rate():>9.2%} "
            f"{row.mean_length():>10.1f}"
        )
