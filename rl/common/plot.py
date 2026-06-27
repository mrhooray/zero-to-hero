from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np


class ReturnSeries(Protocol):
    agent_name: str

    def returns(self) -> np.ndarray: ...


def plot_returns(
    run_groups: list[list[ReturnSeries]],
    output_path: str | Path,
    rolling_window: int = 16,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    show_raw = sum(len(runs) for runs in run_groups) == 1
    for runs in run_groups:
        if show_raw:
            ax.plot(runs[0].returns(), alpha=0.2)
        smoothed_runs = np.stack(
            [_rolling_mean(run.returns(), rolling_window) for run in runs]
        )
        mean = smoothed_runs.mean(axis=0)
        std = smoothed_runs.std(axis=0)
        offset = len(runs[0].returns()) - len(mean)
        episodes = np.arange(offset, offset + len(mean))
        ax.plot(episodes, mean, label=runs[0].agent_name)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("episode")
    ax.set_ylabel(f"episode return, {rolling_window}-episode rolling mean")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")
