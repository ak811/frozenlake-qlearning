from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from policies import epsilon_greedy_action


@dataclass
class QLearningConfig:
    episodes: int = 500
    max_steps: int = 100
    learning_rate: float = 0.8
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995


def q_update(
    q_table: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    learning_rate: float,
    discount_factor: float,
) -> float:
    old_value = q_table[state, action]
    next_max = float(np.max(q_table[next_state]))
    new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
    q_table[state, action] = new_value
    return new_value


def train_qlearning(env, config: QLearningConfig, rng: np.random.Generator) -> tuple[np.ndarray, Dict]:
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    epsilon = float(config.epsilon_start)
    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    epsilons: List[float] = []

    for _ in range(config.episodes):
        state, _info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < config.max_steps:
            action = epsilon_greedy_action(q_table, int(state), epsilon, rng, n_actions)

            next_state, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

            q_update(
                q_table=q_table,
                state=int(state),
                action=int(action),
                reward=float(reward),
                next_state=int(next_state),
                learning_rate=config.learning_rate,
                discount_factor=config.discount_factor,
            )

            total_reward += float(reward)
            steps += 1
            state = next_state

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        epsilons.append(epsilon)

    history = {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "epsilons": epsilons,
    }
    return q_table, history
