from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from planning.type import TrainingResult


def plot_returns(
    results: list[TrainingResult],
    output_path: str | Path,
    rolling_window: int = 16,
    show_raw: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for result in results:
        returns = result.returns()
        if show_raw:
            ax.plot(returns, alpha=0.2)
        smoothed = _rolling_mean(returns, rolling_window)
        offset = len(returns) - len(smoothed)
        ax.plot(
            np.arange(offset, offset + len(smoothed)), smoothed, label=result.agent_name
        )

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
