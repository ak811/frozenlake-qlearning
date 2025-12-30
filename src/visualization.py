from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def qtable_directions_map(qtable: np.ndarray, map_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute max Q-values per state and best-action arrows.
    """
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)

    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)

    eps = np.finfo(float).eps
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            qtable_directions[idx] = directions[int(val)]
        else:
            qtable_directions[idx] = ""

    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_learning_curves(history: dict, outpath: str | Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    rewards = history["episode_rewards"]
    steps = history["episode_steps"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.plot(range(1, len(rewards) + 1), rewards)

    plt.subplot(1, 2, 2)
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.plot(range(1, len(steps) + 1), steps)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_q_values_map(
    qtable: np.ndarray,
    env,
    map_size: int,
    outpath: str | Path,
    frame_title: Optional[str] = None,
) -> None:
    """
    Save a figure with (1) last rendered frame and (2) heatmap with arrows.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last Step" if frame_title is None else frame_title)

    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\n(arrows represent best action)")

    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
