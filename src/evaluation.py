from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from policies import greedy_action


@dataclass
class EvalConfig:
    episodes: int = 50
    max_steps: int = 100


def run_episode(env, q_table: np.ndarray, max_steps: int, capture_frames: bool = False):
    state, _info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    frames = []

    while not done and steps < max_steps:
        action = greedy_action(q_table, int(state))
        if capture_frames:
            frames.append(env.render())
        next_state, reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)

        total_reward += float(reward)
        steps += 1
        state = next_state

    if capture_frames:
        frames.append(env.render())
    return total_reward, steps, frames


def evaluate_q_table(env, q_table: np.ndarray, config: EvalConfig) -> dict:
    rewards: List[float] = []
    steps_list: List[int] = []

    for _ in range(config.episodes):
        r, s, _frames = run_episode(env, q_table, config.max_steps, capture_frames=False)
        rewards.append(r)
        steps_list.append(s)

    success_rate = float(np.mean([1.0 if r > 0.0 else 0.0 for r in rewards]))
    metrics = {
        "episodes": config.episodes,
        "max_steps": config.max_steps,
        "success_rate": success_rate,
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "rewards": rewards,
        "steps": steps_list,
    }
    return metrics
