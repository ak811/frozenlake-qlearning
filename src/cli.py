from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from envs import FrozenLakeSpec, make_frozenlake_env
from evaluation import EvalConfig, evaluate_q_table, run_episode
from qlearning import QLearningConfig, train_qlearning
from utils import ensure_dir, load_json, load_npy, save_json, save_npy, set_global_seed, utc_timestamp
from visualization import plot_learning_curves, plot_q_values_map


def _write_run_artifacts(
    run_dir: Path,
    q_table: np.ndarray,
    config: Dict[str, Any],
    history: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    save_npy(run_dir / "qtable.npy", q_table)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "history.json", history)
    save_json(run_dir / "metrics.json", metrics)


def cmd_train(args: argparse.Namespace) -> int:
    rng = set_global_seed(args.seed)

    spec = FrozenLakeSpec(
        map_size=args.map_size,
        is_slippery=bool(args.slippery),
        map_seed=args.map_seed,
        render_mode="rgb_array",
        desc=None,
    )
    env, desc = make_frozenlake_env(spec)

    train_cfg = QLearningConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    q_table, history = train_qlearning(env, train_cfg, rng)

    # Evaluate (fresh env instance is fine)
    eval_spec = FrozenLakeSpec(
        map_size=args.map_size,
        is_slippery=bool(args.slippery),
        map_seed=args.map_seed,
        render_mode="rgb_array",
        desc=desc,
    )
    eval_env, _ = make_frozenlake_env(eval_spec)
    eval_cfg = EvalConfig(episodes=args.eval_episodes, max_steps=args.max_steps)
    metrics = evaluate_q_table(eval_env, q_table, eval_cfg)

    # Run directory
    base = Path(args.results_dir)
    run_name = f"{utc_timestamp()}_seed{args.seed}_ms{args.map_size}_slip{int(bool(args.slippery))}"
    run_dir = ensure_dir(base / "runs" / run_name)

    # Save artifacts
    config_blob = {
        "seed": args.seed,
        "env": {
            "map_size": args.map_size,
            "is_slippery": bool(args.slippery),
            "map_seed": args.map_seed,
            "desc": desc,
        },
        "train": asdict(train_cfg),
        "eval": asdict(eval_cfg),
    }

    _write_run_artifacts(run_dir, q_table, config_blob, history, metrics)

    # Save plots
    plot_learning_curves(history, run_dir / "learning_curve.png")

    # For the "Last Step" frame: capture one greedy episode frame at the end
    _r, _s, frames = run_episode(eval_env, q_table, args.max_steps, capture_frames=True)
    # eval_env.render() returns current frame; last call already saved in frames[-1]
    # We'll just ensure env is at last rendered state by re-rendering once.
    plot_q_values_map(q_table, eval_env, args.map_size, run_dir / "policy_heatmap.png", frame_title="Final Policy")

    print(f"Saved run to: {run_dir}")
    print(f"Success rate: {metrics['success_rate']:.3f} | Avg steps: {metrics['avg_steps']:.1f}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        config = load_json(run_dir / "config.json")
        q_table = load_npy(run_dir / "qtable.npy")

        env_cfg = config["env"]
        spec = FrozenLakeSpec(
            map_size=int(env_cfg["map_size"]),
            is_slippery=bool(env_cfg["is_slippery"]),
            map_seed=int(env_cfg["map_seed"]),
            render_mode="rgb_array",
            desc=env_cfg.get("desc"),
        )
        env, _ = make_frozenlake_env(spec)
        eval_cfg = EvalConfig(episodes=args.episodes, max_steps=args.max_steps)
        metrics = evaluate_q_table(env, q_table, eval_cfg)
        save_json(run_dir / "metrics_eval.json", metrics)

        print(f"Evaluated run: {run_dir}")
        print(f"Success rate: {metrics['success_rate']:.3f} | Avg steps: {metrics['avg_steps']:.1f}")
        return 0

    # Otherwise use explicit paths and env args
    if not args.qtable:
        raise SystemExit("Provide --run-dir or --qtable")

    q_table = load_npy(args.qtable)

    spec = FrozenLakeSpec(
        map_size=args.map_size,
        is_slippery=bool(args.slippery),
        map_seed=args.map_seed,
        render_mode="rgb_array",
        desc=None,
    )
    env, _desc = make_frozenlake_env(spec)
    eval_cfg = EvalConfig(episodes=args.episodes, max_steps=args.max_steps)
    metrics = evaluate_q_table(env, q_table, eval_cfg)

    print(f"Success rate: {metrics['success_rate']:.3f} | Avg steps: {metrics['avg_steps']:.1f}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / "config.json")
    history = load_json(run_dir / "history.json")
    q_table = load_npy(run_dir / "qtable.npy")

    env_cfg = config["env"]
    spec = FrozenLakeSpec(
        map_size=int(env_cfg["map_size"]),
        is_slippery=bool(env_cfg["is_slippery"]),
        map_seed=int(env_cfg["map_seed"]),
        render_mode="rgb_array",
        desc=env_cfg.get("desc"),
    )
    env, _ = make_frozenlake_env(spec)

    plot_learning_curves(history, run_dir / "learning_curve.png")
    plot_q_values_map(q_table, env, int(env_cfg["map_size"]), run_dir / "policy_heatmap.png", frame_title="Final Policy")

    print(f"Updated plots in: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="frozenlake-qlearning")
    sub = p.add_subparsers(dest="command", required=True)

    # train
    t = sub.add_parser("train", help="Train Q-learning on FrozenLake")
    t.add_argument("--episodes", type=int, default=500)
    t.add_argument("--max-steps", type=int, default=100)

    t.add_argument("--lr", type=float, default=0.8)
    t.add_argument("--gamma", type=float, default=0.95)
    t.add_argument("--epsilon-start", type=float, default=1.0)
    t.add_argument("--epsilon-min", type=float, default=0.01)
    t.add_argument("--epsilon-decay", type=float, default=0.995)

    t.add_argument("--map-size", type=int, default=4)
    t.add_argument("--map-seed", type=int, default=10)
    t.add_argument("--slippery", action="store_true", help="Use slippery dynamics (stochastic transitions)")
    t.add_argument("--seed", type=int, default=0, help="RNG seed for agent/action selection")

    t.add_argument("--eval-episodes", type=int, default=50)
    t.add_argument("--results-dir", type=str, default="results")
    t.set_defaults(func=cmd_train)

    # eval
    e = sub.add_parser("eval", help="Evaluate a saved Q-table")
    e.add_argument("--run-dir", type=str, default=None, help="Evaluate using a run directory (preferred)")
    e.add_argument("--qtable", type=str, default=None, help="Path to qtable.npy if not using --run-dir")

    e.add_argument("--episodes", type=int, default=50)
    e.add_argument("--max-steps", type=int, default=100)

    e.add_argument("--map-size", type=int, default=4)
    e.add_argument("--map-seed", type=int, default=10)
    e.add_argument("--slippery", action="store_true")
    e.set_defaults(func=cmd_eval)

    # plot
    pl = sub.add_parser("plot", help="Regenerate plots for a run directory")
    pl.add_argument("--run-dir", type=str, required=True)
    pl.set_defaults(func=cmd_plot)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
