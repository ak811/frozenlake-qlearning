import numpy as np

from envs import FrozenLakeSpec, make_frozenlake_env
from evaluation import EvalConfig, evaluate_q_table
from qlearning import QLearningConfig, train_qlearning
from utils import set_global_seed


def test_smoke_train_non_slippery_learns_basic_map():
    # Use a fixed, known-solvable map for stability in CI.
    desc = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ]
    spec = FrozenLakeSpec(map_size=4, is_slippery=False, map_seed=0, render_mode="rgb_array", desc=desc)
    env, _ = make_frozenlake_env(spec)

    cfg = QLearningConfig(
        episodes=300,
        max_steps=100,
        learning_rate=0.8,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )
    rng = set_global_seed(0)
    q_table, _history = train_qlearning(env, cfg, rng)

    eval_env, _ = make_frozenlake_env(spec)
    metrics = evaluate_q_table(eval_env, q_table, EvalConfig(episodes=50, max_steps=100))

    # Non-slippery FrozenLake on this map should become reliable quickly.
    assert metrics["success_rate"] >= 0.7
