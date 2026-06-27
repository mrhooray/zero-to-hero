from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from common.grid_runner import evaluate, train
from common.grid_world import GridWorldEnv
from common.plot import plot_returns
from common.seed import spawn_seeds
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
    add_episode_args(train_parser)
    train_parser.add_argument("--agent-seed", type=int, default=24)
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
    add_episode_args(benchmark_parser)
    benchmark_parser.add_argument("--size", type=int, default=8)
    benchmark_parser.add_argument("--seed", type=int, default=24)
    benchmark_parser.add_argument("--runs", type=int, default=8)
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
    env = make_env(args)
    env.reset()
    print(env.render())


def _train(
    args: argparse.Namespace,
    config_fn: Callable[[argparse.Namespace], object],
    agent_map: dict[str, type],
    extra_train_fn: Callable[[argparse.Namespace, object, GridWorldEnv], None] | None,
) -> None:
    config = config_fn(args)
    env = make_env(args)
    agent = agent_map[args.algorithm](env, config)
    train_result = train(env, agent, episodes=args.train_episodes)
    eval_result = evaluate(make_env(args), agent, episodes=args.eval_episodes)

    print(
        f"trained {train_result.agent_name} for {len(train_result.episodes)} episodes"
    )
    print(f"train success rate: {train_result.success_rate():.2%}")
    print(f"eval mean return: {eval_result.mean_return():.3f}")
    print(f"eval success rate: {eval_result.success_rate():.2%}")
    if args.plot:
        plot_returns([[train_result]], args.plot)
        print(f"learning curves plot saved to {args.plot}")
    if extra_train_fn:
        extra_train_fn(args, agent, env)


def _benchmark(
    args: argparse.Namespace,
    config_fn: Callable[[argparse.Namespace], object],
    agent_map: dict[str, type],
    agent_names: tuple[str, ...] | list[str],
) -> None:
    run_seeds = spawn_seeds(args.seed, args.runs)
    benchmark_results: list[list[RunResult]] = []
    row_format = "{:<16}  {:>13}  {:>14}  {:>14}"
    print(f"{'run':>3}  {'env_seed':>10}  {'agent_seed':>10}")
    for run, (env_seed, agent_seed) in enumerate(run_seeds):
        print(f"{run:>3}  {env_seed:>10}  {agent_seed:>10}")
    print()
    print(row_format.format("agent", "mean_return", "success", "mean_len"))
    for name in agent_names:
        train_results = []
        eval_results = []
        for env_seed, agent_seed in run_seeds:
            run_args = argparse.Namespace(
                **{
                    **vars(args),
                    "env_seed": env_seed,
                    "agent_seed": agent_seed,
                }
            )
            env = make_env(run_args)
            agent = agent_map[name](env, config_fn(run_args))
            train_results.append(train(env, agent, episodes=args.train_episodes))
            eval_result = evaluate(
                make_env(run_args),
                agent,
                episodes=args.eval_episodes,
            )
            eval_results.append(eval_result)

        benchmark_results.append(train_results)
        print(
            row_format.format(
                name,
                _mean_std(eval_results, "mean_return"),
                _mean_std(eval_results, "success_rate", percent=True),
                _mean_std(eval_results, "mean_length"),
            )
        )
    if args.plot:
        plot_returns(benchmark_results, args.plot)
        print(f"learning curves plot saved to {args.plot}")


# -------------------------------------------------------------------------
# Argument helpers
# -------------------------------------------------------------------------
def add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--env-seed", type=int, default=24)


def add_episode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-episodes", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=32)


def make_env(args: argparse.Namespace) -> GridWorldEnv:
    start = (0, 0)
    goal = (args.size - 1, args.size - 1)
    return GridWorldEnv(
        size=args.size,
        start=start,
        traps=GridWorldEnv.generate_traps(args.size, args.env_seed, start, goal),
    )


def _mean_std(
    results: list[RunResult],
    metric: str,
    percent: bool = False,
) -> str:
    values = np.array([getattr(result, metric)() for result in results])
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{np.mean(values) * scale:.1f} ± {np.std(values) * scale:.1f}{suffix}"
