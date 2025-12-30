from __future__ import annotations

from typing import Optional

import numpy as np


def greedy_action(q_table: np.ndarray, state: int) -> int:
    return int(np.argmax(q_table[state]))


def epsilon_greedy_action(
    q_table: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
    n_actions: int,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return greedy_action(q_table, state)
